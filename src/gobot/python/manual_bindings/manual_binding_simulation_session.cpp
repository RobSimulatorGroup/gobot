#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gobot/simulation/simulation_session.hpp"

namespace gobot::python {
namespace py = pybind11;
namespace {
class PySimulationExecutor : public SimulationExecutor {
public:
    using Results = std::vector<SimulationMicrostepResult>;
    Results Advance(double dt) override {
        PYBIND11_OVERRIDE_PURE_NAME(Results, SimulationExecutor, "advance", Advance, dt);
    }
    void Reset(const std::vector<std::size_t>& environments) override {
        PYBIND11_OVERRIDE_PURE_NAME(void, SimulationExecutor, "reset", Reset, environments);
    }
    void Close() override {
        PYBIND11_OVERRIDE_PURE_NAME(void, SimulationExecutor, "close", Close);
    }
};
}

void RegisterSimulationSessionBindings(py::module_& module) {
    py::class_<SimulationMicrostepResult>(module, "SimulationMicrostepResult")
            .def(py::init<>())
            .def_readwrite("completed", &SimulationMicrostepResult::completed)
            .def_readwrite("advanced_time", &SimulationMicrostepResult::advanced_time)
            .def_readwrite("state_valid", &SimulationMicrostepResult::state_valid)
            .def_readwrite("requires_reset", &SimulationMicrostepResult::requires_reset)
            .def_readwrite("error", &SimulationMicrostepResult::error)
            .def_readwrite("failure_stage", &SimulationMicrostepResult::failure_stage)
            .def_readwrite("failing_shard", &SimulationMicrostepResult::failing_shard);
    py::class_<SimulationEnvironmentProgress>(module, "SimulationEnvironmentProgress")
            .def_readonly("completed_microsteps", &SimulationEnvironmentProgress::completed_microsteps)
            .def_readonly("completed_ticks", &SimulationEnvironmentProgress::completed_ticks)
            .def_readonly("advanced_time", &SimulationEnvironmentProgress::advanced_time)
            .def_readonly("completed", &SimulationEnvironmentProgress::completed)
            .def_readonly("state_valid", &SimulationEnvironmentProgress::state_valid)
            .def_readonly("requires_reset", &SimulationEnvironmentProgress::requires_reset)
            .def_readonly("error", &SimulationEnvironmentProgress::error)
            .def_readonly("failure_stage", &SimulationEnvironmentProgress::failure_stage)
            .def_readonly("failing_shard", &SimulationEnvironmentProgress::failing_shard);
    py::class_<SimulationStepResult>(module, "SimulationStepResult")
            .def_property_readonly("completed", &SimulationStepResult::Completed)
            .def_readonly("environments", &SimulationStepResult::environments);
    py::class_<SimulationEnvironmentClock>(module, "SimulationEnvironmentClock")
            .def_readonly("episode", &SimulationEnvironmentClock::episode)
            .def_readonly("tick", &SimulationEnvironmentClock::tick)
            .def_readonly("microstep", &SimulationEnvironmentClock::microstep)
            .def_readonly("time", &SimulationEnvironmentClock::time)
            .def_readonly("faulted", &SimulationEnvironmentClock::faulted);
    py::class_<SimulationExecutor, PySimulationExecutor, std::shared_ptr<SimulationExecutor>>(
            module, "_SimulationExecutor").def(py::init<>());
    py::class_<SimulationSession>(module, "_SimulationSession")
            .def(py::init<std::shared_ptr<SimulationExecutor>, std::size_t, double, std::uint32_t>(),
                 py::arg("executor"), py::arg("environment_count"), py::arg("microstep_dt"),
                 py::arg("substeps") = 1, py::keep_alive<1, 2>())
            .def("step", [](SimulationSession& session, std::uint64_t ticks) {
                return session.Step({ticks});
            }, py::arg("ticks") = 1, py::call_guard<py::gil_scoped_release>())
            .def("reset", &SimulationSession::Reset, py::arg("environments") = std::vector<std::size_t>{},
                 py::call_guard<py::gil_scoped_release>())
            .def("close", &SimulationSession::Close, py::call_guard<py::gil_scoped_release>())
            .def_property_readonly("clocks", [](const SimulationSession& session) {
                return session.GetClocks();
            })
            .def_property_readonly("fixed_time_step", &SimulationSession::GetFixedTimeStep)
            .def_property_readonly("closed", &SimulationSession::IsClosed);
}
} // namespace gobot::python
