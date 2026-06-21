#include "manual_bindings_internal.hpp"

#ifdef GOBOT_HAS_MUJOCO
#include <mujoco/mujoco.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#endif

namespace gobot::python {

#ifdef GOBOT_HAS_MUJOCO

namespace {

using BatchClock = std::chrono::steady_clock;
using BatchTimePoint = BatchClock::time_point;

enum BatchProfileIndex : std::size_t {
    kBatchProfileTotal = 0,
    kBatchProfileOutputAlloc,
    kBatchProfileKernelTotal,
    kBatchProfileStateLoad,
    kBatchProfileApplyControl,
    kBatchProfileMjStep,
    kBatchProfileStateStore,
    kBatchProfileSensorCopy,
    kBatchProfileCount,
};

constexpr std::array<const char*, kBatchProfileCount> kBatchProfileNames = {
        "total_ms",
        "output_alloc_ms",
        "kernel_total_ms",
        "state_load_ms",
        "apply_ctrl_ms",
        "mj_step_ms",
        "state_store_ms",
        "sensor_copy_ms",
};

double ElapsedMs(BatchTimePoint begin) {
    return std::chrono::duration<double, std::milli>(BatchClock::now() - begin).count();
}

template <typename T>
py::array_t<T> MakeOwnedArray(std::vector<T> data, std::vector<py::ssize_t> shape) {
    auto* storage = new std::vector<T>(std::move(data));
    py::capsule owner(storage, [](void* pointer) {
        delete static_cast<std::vector<T>*>(pointer);
    });
    T* pointer = storage->empty() ? nullptr : storage->data();
    return py::array_t<T>(std::move(shape), pointer, owner);
}

std::size_t ResolveWorkerCount(std::size_t requested, std::size_t environment_count) {
    if (environment_count == 0) {
        return 0;
    }
    if (requested == 0) {
        requested = std::thread::hardware_concurrency();
    }
    if (requested == 0) {
        requested = 1;
    }
    return std::max<std::size_t>(1, std::min(requested, environment_count));
}

void RequireRankAndShape(const py::buffer_info& info,
                         const std::string& name,
                         std::initializer_list<py::ssize_t> expected) {
    if (info.ndim != static_cast<py::ssize_t>(expected.size())) {
        throw py::value_error(name + " must have rank " + std::to_string(expected.size()));
    }

    std::size_t index = 0;
    for (py::ssize_t size : expected) {
        if (info.shape[static_cast<py::ssize_t>(index)] != size) {
            std::ostringstream stream;
            stream << name << " shape mismatch at axis " << index << ": expected " << size << ", got "
                   << info.shape[static_cast<py::ssize_t>(index)];
            throw py::value_error(stream.str());
        }
        index++;
    }
}

class PyMujocoBatchPool {
public:
    PyMujocoBatchPool(const std::string& xml_path,
                      std::size_t num_envs,
                      std::size_t threads,
                      double timestep)
        : num_envs_(num_envs),
          threads_(ResolveWorkerCount(threads, num_envs)) {
        if (num_envs_ == 0) {
            throw py::value_error("num_envs must be positive");
        }

        char error[1024] = {};
        model_ = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
        if (model_ == nullptr) {
            std::string message = error[0] != '\0' ? std::string(error) : "unknown MuJoCo XML load error";
            throw std::runtime_error("failed to load MuJoCo XML '" + xml_path + "': " + message);
        }
        if (timestep > 0.0) {
            model_->opt.timestep = static_cast<mjtNum>(timestep);
        }

        nstate_ = mj_stateSize(model_, mjSTATE_FULLPHYSICS);
        ncontrol_ = mj_stateSize(model_, mjSTATE_CTRL);
        nsensordata_ = model_->nsensordata;
        const std::size_t data_count = std::max<std::size_t>(1, threads_);
        datas_.reserve(data_count);
        for (std::size_t i = 0; i < data_count; ++i) {
            mjData* data = mj_makeData(model_);
            if (data == nullptr) {
                throw std::runtime_error("failed to allocate MuJoCo data for batch environment");
            }
            datas_.push_back(data);
        }

        if (threads_ > 1) {
            workers_.reserve(threads_);
            for (std::size_t worker_index = 0; worker_index < threads_; ++worker_index) {
                workers_.emplace_back(&PyMujocoBatchPool::WorkerLoop, this, worker_index);
            }
        }
    }

    ~PyMujocoBatchPool() {
        StopWorkers();
        for (mjData* data : datas_) {
            mj_deleteData(data);
        }
        datas_.clear();
        mj_deleteModel(model_);
        model_ = nullptr;
    }

    PyMujocoBatchPool(const PyMujocoBatchPool&) = delete;
    PyMujocoBatchPool& operator=(const PyMujocoBatchPool&) = delete;

    std::size_t GetNumEnvs() const {
        return num_envs_;
    }

    std::size_t GetThreads() const {
        return threads_;
    }

    int GetNq() const {
        return model_->nq;
    }

    int GetNv() const {
        return model_->nv;
    }

    int GetNu() const {
        return model_->nu;
    }

    int GetNstate() const {
        return nstate_;
    }

    int GetNcontrol() const {
        return ncontrol_;
    }

    int GetNsensordata() const {
        return nsensordata_;
    }

    py::dict StepProfile() const {
        py::dict result;
        for (std::size_t index = 0; index < kBatchProfileCount; ++index) {
            result[py::str(kBatchProfileNames[index])] = last_step_profile_ms_[index];
        }
        return result;
    }

    py::array_t<double> InitialState() {
        std::vector<double> output(num_envs_ * static_cast<std::size_t>(nstate_));
        std::vector<mjtNum> state(static_cast<std::size_t>(nstate_));
        mjData* data = datas_[0];
        mj_resetData(model_, data);
        mj_forward(model_, data);
        mj_getState(model_, data, state.data(), mjSTATE_FULLPHYSICS);
        for (std::size_t env = 0; env < num_envs_; ++env) {
            std::memcpy(output.data() + env * static_cast<std::size_t>(nstate_),
                        state.data(),
                        static_cast<std::size_t>(nstate_) * sizeof(double));
        }
        return MakeOwnedArray<double>(std::move(output),
                                      {static_cast<py::ssize_t>(num_envs_), static_cast<py::ssize_t>(nstate_)});
    }

    py::object Step(py::array_t<double, py::array::c_style | py::array::forcecast> state0,
                    py::object control,
                    int nstep,
                    bool return_sensor) {
        const BatchTimePoint total_begin = BatchClock::now();
        if (nstep <= 0) {
            throw py::value_error("nstep must be positive");
        }

        py::buffer_info state_info = state0.request();
        RequireRankAndShape(state_info,
                            "state0",
                            {static_cast<py::ssize_t>(num_envs_), static_cast<py::ssize_t>(nstate_)});
        const auto* state_ptr = static_cast<const double*>(state_info.ptr);

        py::array_t<double, py::array::c_style | py::array::forcecast> control_array;
        const double* control_ptr = nullptr;
        if (!control.is_none()) {
            control_array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(control);
            if (!control_array) {
                throw py::value_error("control must be convertible to a contiguous float64 ndarray");
            }
            py::buffer_info control_info = control_array.request();
            RequireRankAndShape(control_info,
                                "control",
                                {static_cast<py::ssize_t>(num_envs_),
                                 static_cast<py::ssize_t>(nstep),
                                 static_cast<py::ssize_t>(ncontrol_)});
            control_ptr = static_cast<const double*>(control_info.ptr);
        }

        const std::size_t state_width = static_cast<std::size_t>(nstate_);
        const std::size_t control_width = static_cast<std::size_t>(ncontrol_);
        const std::size_t sensor_width = static_cast<std::size_t>(std::max(0, nsensordata_));
        std::array<double, kBatchProfileCount> profile_values{};
        BatchTimePoint phase_begin = BatchClock::now();
        std::vector<double> state_out(num_envs_ * state_width);
        std::vector<double> sensor_out;
        if (return_sensor) {
            sensor_out.resize(num_envs_ * sensor_width);
        }
        profile_values[kBatchProfileOutputAlloc] = ElapsedMs(phase_begin);

        std::vector<std::array<double, kBatchProfileCount>> worker_profiles(threads_);
        for (auto& profile : worker_profiles) {
            profile.fill(0.0);
        }

        phase_begin = BatchClock::now();
        {
            py::gil_scoped_release release;
            RunParallel([&](std::size_t worker_index, std::size_t begin, std::size_t end) {
                auto& worker_profile = worker_profiles[std::min(worker_index, worker_profiles.size() - 1)];
                mjData* data = datas_[worker_index];
                for (std::size_t env = begin; env < end; ++env) {
                    BatchTimePoint step_begin = BatchClock::now();
                    mj_setState(model_, data, state_ptr + env * state_width, mjSTATE_FULLPHYSICS);
                    if (model_->nv > 0) {
                        mju_zero(data->qacc_warmstart, model_->nv);
                    }
                    worker_profile[kBatchProfileStateLoad] += ElapsedMs(step_begin);
                    step_begin = BatchClock::now();
                    for (int step = 0; step < nstep; ++step) {
                        if (model_->nu > 0) {
                            if (control_ptr != nullptr) {
                                const double* control_step =
                                        control_ptr + (env * static_cast<std::size_t>(nstep) +
                                                       static_cast<std::size_t>(step)) *
                                                              control_width;
                                mju_copy(data->ctrl, control_step, model_->nu);
                            } else {
                                mju_zero(data->ctrl, model_->nu);
                            }
                        }
                        mj_step(model_, data);
                    }
                    worker_profile[kBatchProfileMjStep] += ElapsedMs(step_begin);
                    step_begin = BatchClock::now();
                    mj_getState(model_, data, state_out.data() + env * state_width, mjSTATE_FULLPHYSICS);
                    worker_profile[kBatchProfileStateStore] += ElapsedMs(step_begin);
                    if (return_sensor && sensor_width > 0) {
                        step_begin = BatchClock::now();
                        std::memcpy(sensor_out.data() + env * sensor_width,
                                    data->sensordata,
                                    sensor_width * sizeof(double));
                        worker_profile[kBatchProfileSensorCopy] += ElapsedMs(step_begin);
                    }
                }
            });
        }
        profile_values[kBatchProfileKernelTotal] = ElapsedMs(phase_begin);
        for (std::size_t profile_index = kBatchProfileStateLoad; profile_index < kBatchProfileCount; ++profile_index) {
            double critical_path_ms = 0.0;
            for (const auto& worker_profile : worker_profiles) {
                critical_path_ms = std::max(critical_path_ms, worker_profile[profile_index]);
            }
            profile_values[profile_index] = critical_path_ms;
        }

        py::array_t<double> state_array =
                MakeOwnedArray<double>(std::move(state_out),
                                       {static_cast<py::ssize_t>(num_envs_), static_cast<py::ssize_t>(nstate_)});
        profile_values[kBatchProfileTotal] = ElapsedMs(total_begin);
        last_step_profile_ms_ = profile_values;
        if (!return_sensor) {
            return state_array;
        }

        py::array_t<double> sensor_array =
                MakeOwnedArray<double>(std::move(sensor_out),
                                       {static_cast<py::ssize_t>(num_envs_),
                                        static_cast<py::ssize_t>(nsensordata_)});
        return py::make_tuple(state_array, sensor_array);
    }

private:
    using BatchTask = std::function<void(std::size_t, std::size_t, std::size_t)>;

    void RunParallel(BatchTask func) {
        if (workers_.empty() || num_envs_ <= 1) {
            func(0, 0, num_envs_);
            return;
        }

        {
            std::unique_lock lock(task_mutex_);
            task_ = std::move(func);
            finished_workers_ = 0;
            task_chunk_size_ = std::max<std::size_t>(1, num_envs_ / (workers_.size() * 10));
            next_env_.store(0);
            task_generation_++;
        }
        task_cv_.notify_all();

        std::unique_lock lock(task_mutex_);
        done_cv_.wait(lock, [this]() {
            return finished_workers_ == workers_.size();
        });
        task_ = nullptr;
    }

    void WorkerLoop(std::size_t worker_index) {
        std::size_t observed_generation = 0;
        while (true) {
            BatchTask task;
            std::size_t generation = 0;
            {
                std::unique_lock lock(task_mutex_);
                task_cv_.wait(lock, [this, observed_generation]() {
                    return stop_workers_ || task_generation_ != observed_generation;
                });
                if (stop_workers_) {
                    return;
                }
                task = task_;
                generation = task_generation_;
            }

            while (true) {
                const std::size_t begin = next_env_.fetch_add(task_chunk_size_);
                if (begin >= num_envs_) {
                    break;
                }
                const std::size_t end = std::min(num_envs_, begin + task_chunk_size_);
                task(worker_index, begin, end);
            }

            {
                std::lock_guard lock(task_mutex_);
                observed_generation = generation;
                finished_workers_++;
                if (finished_workers_ == workers_.size()) {
                    done_cv_.notify_one();
                }
            }
        }
    }

    void StopWorkers() {
        {
            std::lock_guard lock(task_mutex_);
            stop_workers_ = true;
        }
        task_cv_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }

    mjModel* model_{nullptr};
    std::vector<mjData*> datas_;
    std::vector<std::thread> workers_;
    std::mutex task_mutex_;
    std::condition_variable task_cv_;
    std::condition_variable done_cv_;
    BatchTask task_;
    std::atomic<std::size_t> next_env_{0};
    std::size_t task_chunk_size_{1};
    std::size_t task_generation_{0};
    std::size_t finished_workers_{0};
    bool stop_workers_{false};
    std::size_t num_envs_{0};
    std::size_t threads_{1};
    int nstate_{0};
    int ncontrol_{0};
    int nsensordata_{0};
    std::array<double, kBatchProfileCount> last_step_profile_ms_{};
};

} // namespace

#endif

void RegisterManualMujocoBatchBindings(py::module_& module) {
#ifdef GOBOT_HAS_MUJOCO
    module.attr("_has_mujoco_batch_pool") = true;
    py::class_<PyMujocoBatchPool>(module, "_MujocoBatchPool")
            .def(py::init<const std::string&, std::size_t, std::size_t, double>(),
                 py::arg("xml_path"),
                 py::arg("num_envs"),
                 py::arg("threads") = 0,
                 py::arg("timestep") = 0.002)
            .def_property_readonly("num_envs", &PyMujocoBatchPool::GetNumEnvs)
            .def_property_readonly("threads", &PyMujocoBatchPool::GetThreads)
            .def_property_readonly("nq", &PyMujocoBatchPool::GetNq)
            .def_property_readonly("nv", &PyMujocoBatchPool::GetNv)
            .def_property_readonly("nu", &PyMujocoBatchPool::GetNu)
            .def_property_readonly("nstate", &PyMujocoBatchPool::GetNstate)
            .def_property_readonly("ncontrol", &PyMujocoBatchPool::GetNcontrol)
            .def_property_readonly("nsensordata", &PyMujocoBatchPool::GetNsensordata)
            .def("initial_state", &PyMujocoBatchPool::InitialState)
            .def("step_profile", &PyMujocoBatchPool::StepProfile)
            .def("step",
                 &PyMujocoBatchPool::Step,
                 py::arg("state0"),
                 py::arg("control") = py::none(),
                 py::arg("nstep") = 1,
                 py::arg("return_sensor") = false);
#else
    module.attr("_has_mujoco_batch_pool") = false;
#endif
}

} // namespace gobot::python
