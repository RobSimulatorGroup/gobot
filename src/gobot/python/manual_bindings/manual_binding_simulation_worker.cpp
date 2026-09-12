#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gobot/simulation/simulation_data_runtime.hpp"

namespace gobot::python {
namespace py = pybind11;
namespace {
std::vector<SimulationBuffer> CopyBuffers(const py::dict& mapping) {
    std::vector<SimulationBuffer> buffers;
    for (auto item : mapping) {
        auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(item.second);
        if (!array) throw std::invalid_argument("Simulation fields must be numeric arrays");
        SimulationBuffer buffer;
        buffer.name = py::cast<std::string>(item.first);
        for (auto size : array.request().shape) buffer.shape.push_back(static_cast<std::size_t>(size));
        buffer.values.assign(array.data(), array.data() + array.size());
        if (!std::all_of(buffer.values.begin(), buffer.values.end(), [](double v) { return std::isfinite(v); }))
            throw std::invalid_argument("Simulation fields must contain finite values");
        buffers.push_back(std::move(buffer));
    }
    return buffers;
}

class PythonDataExecutor final : public SimulationDataExecutor {
public:
    explicit PythonDataExecutor(py::object adapter) : adapter_(std::move(adapter)) {}
    ~PythonDataExecutor() override {
        py::gil_scoped_acquire gil;
        adapter_ = py::object{};
    }
    std::vector<SimulationMicrostepResult> Advance(double dt) override {
        py::gil_scoped_acquire gil;
        return adapter_.attr("advance")(dt).cast<std::vector<SimulationMicrostepResult>>();
    }
    void Reset(const std::vector<std::size_t>& indices) override {
        py::gil_scoped_acquire gil;
        adapter_.attr("reset")(indices);
    }
    void Close() override {
        py::gil_scoped_acquire gil;
        adapter_.attr("close")();
    }
    void ApplyCommands(const std::vector<SimulationBuffer>& commands) override {
        py::gil_scoped_acquire gil;
        py::dict mapping;
        for (const auto& command : commands) {
            std::vector<py::ssize_t> shape(command.shape.begin(), command.shape.end());
            py::array_t<double> array(shape);
            if (!command.values.empty()) std::memcpy(array.mutable_data(), command.values.data(),
                    command.values.size() * sizeof(double));
            mapping[py::str(command.name)] = std::move(array);
        }
        adapter_.attr("apply_commands")(mapping);
    }
    void CaptureSnapshot(const SimulationSubscription& subscription,
                         std::vector<SimulationBuffer>& buffers) override {
        py::gil_scoped_acquire gil;
        buffers = CopyBuffers(adapter_.attr("snapshot")(subscription.fields, subscription.environments));
    }
private:
    py::object adapter_; // Created, closed and decref'd on the worker, under the GIL.
};

struct SnapshotLease { std::shared_ptr<const SimulationSnapshot> frame; };
struct SnapshotBufferLease {
    std::shared_ptr<const SimulationSnapshot> frame;
    std::size_t index;
};

// Keep one CPython thread state for the native worker's entire lifetime.
// Third-party pybind11 builds (notably Torch) can cache the state used on
// first import in their own TLS. Creating/deleting it for each callback leaves
// those caches dangling. GILState also lets all pybind11 ABIs find this same
// state; the GIL itself remains released between callbacks.
class PythonWorkerThreadState {
public:
    PythonWorkerThreadState() : gil_state_(PyGILState_Ensure()), state_(PyEval_SaveThread()) {}
    ~PythonWorkerThreadState() {
        PyEval_RestoreThread(state_);
        PyGILState_Release(gil_state_);
    }
    PythonWorkerThreadState(const PythonWorkerThreadState&) = delete;
    PythonWorkerThreadState& operator=(const PythonWorkerThreadState&) = delete;
private:
    PyGILState_STATE gil_state_;
    PyThreadState* state_;
};

struct PythonSimulationDataRuntime : SimulationDataRuntime {
    ~PythonSimulationDataRuntime() {
        // Release all Python objects before destroying the persistent state,
        // including after a failed installation. Retire is idempotent.
        Retire();
    }
private:
    PythonWorkerThreadState thread_state_;
};

class PythonSimulationWorker {
public:
    ~PythonSimulationWorker() {
        // Python GC may destroy the last wrapper while its provider needs the
        // GIL to finish/close. The explicit shutdown path uses the same rule.
        if (Py_IsInitialized() && PyGILState_Check()) {
            py::gil_scoped_release release;
            worker.Shutdown();
        } else {
            worker.Shutdown();
        }
    }
    SimulationTaskWorker<PythonSimulationDataRuntime> worker;
};
}

void RegisterSimulationWorkerBindings(py::module_& module) {
    using Runtime = SimulationDataRuntime;
    py::class_<SimulationRuntimeInfo>(module, "SimulationRuntimeInfo")
            .def_readonly("provider_name", &SimulationRuntimeInfo::provider_name)
            .def_readonly("device", &SimulationRuntimeInfo::device)
            .def_readonly("graph_status", &SimulationRuntimeInfo::graph_status)
            .def_readonly("environment_count", &SimulationRuntimeInfo::environment_count)
            .def_readonly("fixed_time_step", &SimulationRuntimeInfo::fixed_time_step)
            .def_readonly("controlled_joint_count", &SimulationRuntimeInfo::controlled_joint_count)
            .def_readonly("capacities", &SimulationRuntimeInfo::capacities);
    py::class_<SnapshotBufferLease>(module, "_SimulationSnapshotBuffer", py::buffer_protocol())
            .def_buffer([](SnapshotBufferLease& lease) {
                const auto& buffer = lease.frame->buffers.at(lease.index);
                std::vector<py::ssize_t> shape(buffer.shape.begin(), buffer.shape.end());
                std::vector<py::ssize_t> strides(shape.size());
                py::ssize_t stride = sizeof(double);
                for (std::size_t i = shape.size(); i > 0; --i) {
                    strides[i - 1] = stride;
                    stride *= shape[i - 1];
                }
                return py::buffer_info(const_cast<double*>(buffer.values.data()), sizeof(double),
                        py::format_descriptor<double>::format(), shape.size(), shape, strides, true);
            });
    py::class_<SnapshotLease>(module, "SimulationSnapshot")
            .def_property_readonly("epoch", [](const SnapshotLease& lease) { return lease.frame->epoch; })
            .def_property_readonly("environments", [](const SnapshotLease& lease) { return lease.frame->environments; })
            .def_property_readonly("clocks", [](const SnapshotLease& lease) { return lease.frame->clocks; })
            .def_property_readonly("fields", [](const SnapshotLease& lease) {
                std::vector<std::string> names;
                for (const auto& buffer : lease.frame->buffers) names.push_back(buffer.name);
                return names;
            })
            .def("buffer", [](const SnapshotLease& lease, const std::string& name) {
                for (std::size_t i = 0; i < lease.frame->buffers.size(); ++i)
                    if (lease.frame->buffers[i].name == name) return SnapshotBufferLease{lease.frame, i};
                throw py::key_error(name);
            });
    py::class_<Runtime::Completion>(module, "SimulationCompletion")
            .def_property_readonly("operation", [](const Runtime::Completion& result) {
                switch (result.operation) {
                    case Runtime::Operation::Install: return "install";
                    case Runtime::Operation::Step: return "step";
                    case Runtime::Operation::Reset: return "reset";
                    case Runtime::Operation::Subscribe: return "subscribe";
                }
                return "unknown";
            })
            .def_readonly("epoch", &Runtime::Completion::epoch)
            .def_readonly("info", &Runtime::Completion::info)
            .def_readonly("step", &Runtime::Completion::step)
            .def_readonly("clocks", &Runtime::Completion::clocks)
            .def_readonly("error", &Runtime::Completion::error)
            .def_readonly("presentation_error", &Runtime::Completion::presentation_error)
            .def_readonly("elapsed_seconds", &Runtime::Completion::elapsed_seconds)
            .def_readonly("snapshot_skipped", &Runtime::Completion::snapshot_skipped)
            .def_property_readonly("snapshot", [](const Runtime::Completion& result) -> std::optional<SnapshotLease> {
                if (!result.snapshot) return std::nullopt;
                return SnapshotLease{result.snapshot};
            });
    py::class_<PythonSimulationWorker>(module, "_SimulationDataWorker")
            .def(py::init<>())
            .def("install", [](PythonSimulationWorker& self, std::string recipe, std::uint64_t epoch,
                               std::vector<std::string> fields, std::vector<std::size_t> environments,
                               double interval) {
                Runtime::Request request{.operation = Runtime::Operation::Install, .epoch = epoch};
                request.subscription = {std::move(fields), std::move(environments), interval};
                request.factory = [recipe = std::move(recipe)] {
                    py::gil_scoped_acquire gil;
                    py::object adapter = py::module_::import("gobot.sim.runtime").attr("_create_executor")(recipe);
                    auto executor = std::make_shared<PythonDataExecutor>(adapter);
                    try {
                        const auto description = adapter.attr("describe")().cast<py::dict>();
                        SimulationRuntimeInfo info;
                        info.provider_name = description["provider_name"].cast<std::string>();
                        info.device = description["device"].cast<std::string>();
                        info.graph_status = description["graph_status"].cast<std::string>();
                        info.controlled_joint_count = description["controlled_joint_count"].cast<std::size_t>();
                        info.capacities = description["capacities"].cast<std::map<std::string, std::uint64_t>>();
                        return SimulationExecutorInstallation{executor,
                                adapter.attr("environment_count").cast<std::size_t>(),
                                adapter.attr("microstep_dt").cast<double>(),
                                adapter.attr("substeps").cast<std::uint32_t>(), std::move(info)};
                    } catch (...) {
                        try { executor->Close(); } catch (...) {}
                        throw;
                    }
                };
                return self.worker.Submit(std::move(request));
            })
            .def("step", [](PythonSimulationWorker& self, const py::dict& commands,
                            std::uint64_t ticks, std::uint64_t epoch) {
                Runtime::Request request{.operation = Runtime::Operation::Step, .epoch = epoch, .ticks = ticks};
                request.commands = CopyBuffers(commands);
                return self.worker.Submit(std::move(request));
            })
            .def("reset", [](PythonSimulationWorker& self, std::vector<std::size_t> environments, std::uint64_t epoch) {
                Runtime::Request request{.operation = Runtime::Operation::Reset, .epoch = epoch};
                request.reset_environments = std::move(environments);
                return self.worker.RequestControl(std::move(request));
            })
            .def("subscribe", [](PythonSimulationWorker& self, std::vector<std::string> fields,
                                 std::vector<std::size_t> environments, double interval, std::uint64_t epoch) {
                Runtime::Request request{.operation = Runtime::Operation::Subscribe, .epoch = epoch};
                request.subscription = {std::move(fields), std::move(environments), interval};
                return self.worker.Submit(std::move(request));
            })
            .def("poll", [](PythonSimulationWorker& self) { return self.worker.Poll(); })
            .def_property_readonly("ready", [](const PythonSimulationWorker& self) { return self.worker.CanSubmit(); })
            .def_property_readonly("pending", [](const PythonSimulationWorker& self) { return self.worker.IsPending(); })
            .def("retire", [](PythonSimulationWorker& self) { self.worker.Retire(); })
            .def("shutdown", [](PythonSimulationWorker& self) { self.worker.Shutdown(); },
                 py::call_guard<py::gil_scoped_release>());
}
} // namespace gobot::python
