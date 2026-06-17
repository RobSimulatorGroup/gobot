#include "manual_bindings_internal.hpp"

namespace gobot::python {

void RegisterManualAppContextBindings(py::module_& module) {
    py::class_<EngineContext, std::shared_ptr<EngineContext>>(module, "AppContext")
            .def_property_readonly("project_path", &EngineContext::GetProjectPath)
            .def_property_readonly("scene_path", &EngineContext::GetScenePath)
            .def_property_readonly("scene_epoch", &EngineContext::GetSceneEpoch)
            .def_property_readonly("scene_dirty", &EngineContext::IsSceneDirty)
            .def_property_readonly("can_undo", &EngineContext::CanUndoSceneCommand)
            .def_property_readonly("can_redo", &EngineContext::CanRedoSceneCommand)
            .def_property_readonly("undo_name", &EngineContext::GetUndoSceneCommandName)
            .def_property_readonly("redo_name", &EngineContext::GetRedoSceneCommandName)
            .def_property_readonly("root", [](EngineContext& context) -> py::object {
                Node* root = SceneRootForContext(context);
                if (root == nullptr) {
                    return py::none();
                }
                return MakeTypedNodeObject(root,
                                           PyNodeOwnership::Borrowed,
                                           &context,
                                           SceneEpochForContext(&context));
            })
            .def_property_readonly("input", [](EngineContext&) -> Input* {
                return Input::GetInstanceOrNull();
            }, py::return_value_policy::reference)
            .def_property("backend_type",
                          &EngineContext::GetBackendType,
                          &EngineContext::SetBackendType)
            .def_property_readonly("has_scene", &EngineContext::HasScene)
            .def_property_readonly("has_world", &EngineContext::HasWorld)
            .def_property_readonly("simulation_time", &EngineContext::GetSimulationTime)
            .def_property_readonly("frame_count", &EngineContext::GetFrameCount)
            .def_property("fixed_time_step",
                          &EngineContext::GetFixedTimeStep,
                          [](EngineContext& context, RealType fixed_time_step) {
                              if (!context.SetFixedTimeStep(fixed_time_step)) {
                                  throw std::runtime_error(context.GetLastError());
                              }
                          })
            .def_property("max_sub_steps",
                          &EngineContext::GetMaxSubSteps,
                          [](EngineContext& context, int max_sub_steps) {
                              if (!context.SetMaxSubSteps(max_sub_steps)) {
                                  throw std::runtime_error(context.GetLastError());
                              }
                          })
            .def_property_readonly("gravity", [](const EngineContext& context) {
                return Vector3ToPython(context.GetGravity());
            })
            .def("set_project_path", [](EngineContext& context, const std::string& project_path) {
                if (!context.SetProjectPath(project_path)) {
                    throw std::runtime_error(context.GetLastError());
                }
            }, py::arg("project_path"))
            .def("load_scene", [](EngineContext& context, const std::string& scene_path) {
                if (!context.LoadScene(scene_path)) {
                    throw std::runtime_error(context.GetLastError());
                }
                return MakeTypedNodeObject(context.GetSceneRoot(),
                                           PyNodeOwnership::Borrowed,
                                           &context,
                                           context.GetSceneEpoch());
            }, py::arg("scene_path"))
            .def("clear_scene", &EngineContext::ClearScene)
            .def("notify_scene_changed", &EngineContext::NotifySceneChanged)
            .def("mark_scene_clean", &EngineContext::MarkSceneClean)
            .def("undo", [](EngineContext& context) {
                return context.UndoSceneCommand();
            })
            .def("redo", [](EngineContext& context) {
                return context.RedoSceneCommand();
            })
            .def("begin_transaction", [](EngineContext& context, const std::string& name) {
                if (!context.BeginSceneTransaction(name)) {
                    throw std::runtime_error("failed to begin Gobot scene transaction '" + name + "'");
                }
            }, py::arg("name") = "Scene Transaction")
            .def("commit_transaction", [](EngineContext& context) {
                if (!context.CommitSceneTransaction()) {
                    throw std::runtime_error("failed to commit Gobot scene transaction");
                }
            })
            .def("cancel_transaction", [](EngineContext& context) {
                if (!context.CancelSceneTransaction()) {
                    throw std::runtime_error("failed to cancel Gobot scene transaction");
                }
            })
            .def("transaction", [](EngineContext&, const std::string& name) {
                return PySceneTransaction(name);
            }, py::arg("name") = "Scene Transaction")
            .def("build_world", [](EngineContext& context, PhysicsBackendType backend_type) {
                context.SetBackendType(backend_type);
                if (!context.BuildWorld()) {
                    throw std::runtime_error(context.GetLastError());
                }
            }, py::arg("backend_type") = PhysicsBackendType::Null)
            .def("rebuild_world", [](EngineContext& context, bool preserve_state) {
                if (!context.RebuildWorld(preserve_state)) {
                    throw std::runtime_error(context.GetLastError());
                }
            }, py::arg("preserve_state") = true)
            .def("clear_world", &EngineContext::ClearWorld)
            .def("reset_simulation", [](EngineContext& context) {
                if (!context.ResetSimulation()) {
                    throw std::runtime_error(context.GetLastError());
                }
            })
            .def("step_once", [](EngineContext& context) {
                if (!context.StepOnce()) {
                    throw std::runtime_error(context.GetLastError());
                }
            })
            .def("step", [](EngineContext& context, std::uint64_t ticks) {
                if (!context.StepTicks(ticks)) {
                    throw std::runtime_error(context.GetLastError());
                }
            }, py::arg("ticks") = 1)
            .def("configure_batch_world", [](EngineContext& context, std::size_t num_envs) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                if (!simulation->ConfigureEnvironmentBatch(num_envs)) {
                    throw std::runtime_error(simulation->GetLastError());
                }
            }, py::arg("num_envs"))
            .def_property_readonly("batch_env_count", [](EngineContext& context) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    return static_cast<std::size_t>(0);
                }
                return simulation->GetEnvironmentCount();
            })
            .def("reset_batch_env", [](EngineContext& context, std::size_t env_id) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                if (!simulation->ResetEnvironment(env_id)) {
                    throw std::runtime_error(simulation->GetLastError());
                }
            }, py::arg("env_id"))
            .def("step_batch_env", [](EngineContext& context, std::size_t env_id, std::uint64_t ticks) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                if (!simulation->StepEnvironment(env_id, ticks)) {
                    throw std::runtime_error(simulation->GetLastError());
                }
            }, py::arg("env_id"), py::arg("ticks") = 1)
            .def("step_batch", [](EngineContext& context, std::uint64_t ticks, std::size_t workers) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                py::gil_scoped_release release;
                if (!simulation->StepEnvironmentBatch(ticks, workers)) {
                    py::gil_scoped_acquire acquire;
                    throw std::runtime_error(simulation->GetLastError());
                }
            }, py::arg("ticks") = 1, py::arg("workers") = 0)
            .def("resolved_batch_workers", [](EngineContext& context, std::size_t workers) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                return simulation->ResolveEnvironmentBatchWorkerCount(workers);
            }, py::arg("workers") = 0)
            .def("set_batch_joint_position_target", [](EngineContext& context,
                                                       std::size_t env_id,
                                                       const std::string& robot,
                                                       const std::string& joint,
                                                       RealType target_position) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                if (!runtime_scene->SetEnvironmentJointPositionTarget(env_id, robot, joint, target_position)) {
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("env_id"), py::arg("robot"), py::arg("joint"), py::arg("target_position"))
            .def("set_batch_joint_position_targets", [](EngineContext& context,
                                                        const std::string& robot,
                                                        const std::vector<std::string>& joint_names,
                                                        py::array_t<RealType, py::array::c_style | py::array::forcecast> target_positions) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                const py::buffer_info buffer = target_positions.request();
                if (buffer.ndim != 2) {
                    throw std::invalid_argument("target_positions must be a 2D array with shape [num_envs, num_joints]");
                }
                const auto environment_count = static_cast<std::size_t>(buffer.shape[0]);
                const auto joint_count = static_cast<std::size_t>(buffer.shape[1]);
                if (joint_count != joint_names.size()) {
                    throw std::invalid_argument(fmt::format("target_positions has {} joint column(s), expected {}",
                                                            joint_count,
                                                            joint_names.size()));
                }
                const auto* data = static_cast<const RealType*>(buffer.ptr);
                std::vector<RealType> targets(data, data + environment_count * joint_count);
                if (!runtime_scene->SetEnvironmentJointPositionTargets(robot,
                                                                       joint_names,
                                                                       targets,
                                                                       environment_count)) {
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("robot"), py::arg("joint_names"), py::arg("target_positions"))
            .def("reset_batch_joint_state", [](EngineContext& context,
                                               std::size_t env_id,
                                               const std::string& robot,
                                               const std::string& joint,
                                               RealType position,
                                               RealType velocity) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                if (!runtime_scene->ResetEnvironmentJointState(env_id, robot, joint, position, velocity)) {
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("env_id"), py::arg("robot"), py::arg("joint"), py::arg("position"), py::arg("velocity") = 0.0)
            .def("reset_batch_link_state", [](EngineContext& context,
                                              std::size_t env_id,
                                              const std::string& robot,
                                              const std::string& link,
                                              const py::object& position,
                                              const py::object& orientation,
                                              const py::object& linear_velocity,
                                              const py::object& angular_velocity) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                SimulationScene* runtime_scene = simulation->GetRuntimeScene();
                if (runtime_scene == nullptr) {
                    throw std::runtime_error("simulation runtime scene has not been built");
                }
                if (!runtime_scene->ResetEnvironmentLinkState(env_id,
                                                              robot,
                                                              link,
                                                              PythonToVector3(position),
                                                              PythonToQuaternionWxyz(orientation),
                                                              PythonToVector3(linear_velocity),
                                                              PythonToVector3(angular_velocity))) {
                    throw std::runtime_error(runtime_scene->GetLastError());
                }
            }, py::arg("env_id"),
               py::arg("robot"),
               py::arg("link"),
               py::arg("position"),
               py::arg("orientation") = py::make_tuple(1.0, 0.0, 0.0, 0.0),
               py::arg("linear_velocity") = py::make_tuple(0.0, 0.0, 0.0),
               py::arg("angular_velocity") = py::make_tuple(0.0, 0.0, 0.0))
            .def("set_default_joint_gains", [](EngineContext& context, py::dict gains) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                simulation->SetDefaultJointGains(DictToReflected<JointControllerGains>(gains));
            }, py::arg("gains"))
            .def("get_default_joint_gains", [](EngineContext& context) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                return ReflectedToPythonDict(simulation->GetDefaultJointGains());
            })
            .def("get_batch_runtime_state", [](EngineContext& context, std::size_t env_id) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                const PhysicsSceneState* state = simulation->GetEnvironmentState(env_id);
                if (state == nullptr) {
                    throw std::runtime_error("simulation environment state is not available");
                }
                return RuntimeStateToPythonDict(*state);
            }, py::arg("env_id"))
            .def("get_batch_robot_state", [](EngineContext& context,
                                             const std::string& robot,
                                             const std::string& base_link,
                                             const std::vector<std::string>& joint_names,
                                             const std::vector<std::string>& link_names,
                                             const std::vector<std::string>& sensor_names) {
                SimulationServer* simulation = context.GetSimulationServer();
                if (simulation == nullptr) {
                    throw std::runtime_error("active Gobot app context has no SimulationServer");
                }
                return BatchRobotStateToPythonDict(*simulation,
                                                   robot,
                                                   base_link,
                                                   joint_names,
                                                   link_names,
                                                   sensor_names);
            }, py::arg("robot"),
               py::arg("base_link"),
               py::arg("joint_names"),
               py::arg("link_names") = std::vector<std::string>{},
               py::arg("sensor_names") = std::vector<std::string>{})
            .def("get_last_error", &EngineContext::GetLastError);

    py::class_<PyScene, std::unique_ptr<PyScene>>(module, "Scene")
            .def_property_readonly("root", [](PyScene& scene) {
                return MakeTypedNodeObject(scene.root);
            })
            .def_property_readonly("scene_epoch", [](const PyScene& scene) {
                return scene.scene_epoch;
            });

    py::class_<PySceneTransaction>(module, "SceneTransaction")
            .def("__enter__", &PySceneTransaction::Enter, py::return_value_policy::reference_internal)
            .def("__exit__", &PySceneTransaction::Exit,
                 py::arg("exc_type"),
                 py::arg("exc_value"),
                 py::arg("traceback"));
}

} // namespace gobot::python
