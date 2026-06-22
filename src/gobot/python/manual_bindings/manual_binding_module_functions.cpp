#include "manual_bindings_internal.hpp"

#include "../../rl/task_llvm.hpp"

namespace gobot::python {

void RegisterManualModuleFunctions(py::module_& module) {
    module.def("set_project_path", [](const std::string& project_path) {
        EngineContext& context = GetActiveAppContext();
        if (!context.SetProjectPath(project_path)) {
            throw std::runtime_error("failed to set Gobot project path '" + project_path + "'");
        }
    });

    py::module_ app_module = module.def_submodule("app", "Gobot application runtime context.");
    app_module.def("context", []() {
        return std::shared_ptr<EngineContext>(&GetActiveAppContext(), [](EngineContext*) {});
    });
    app_module.def("create_context", []() {
        return CreateAppContext();
    });

    module.def("load_scene", [](const std::string& scene_path) {
        EnsureRuntimeContext();
        return std::make_unique<PyScene>(LoadSceneRoot(scene_path));
    }, py::arg("scene_path"));

    module.def("create_node", [](const std::string& type_name, const std::string& name) {
        EnsureRuntimeContext();
        return MakeTypedNodeObject(CreateNode(type_name, name), PyNodeOwnership::DetachedOwned);
    }, py::arg("type_name"), py::arg("name") = "");

    module.def("transaction", [](const std::string& name) {
        EnsureRuntimeContext();
        return PySceneTransaction(name);
    }, py::arg("name") = "Scene Transaction");

    module.def("undo", []() {
        EnsureRuntimeContext();
        return GetActiveAppContext().UndoSceneCommand();
    });

    module.def("redo", []() {
        EnsureRuntimeContext();
        return GetActiveAppContext().RedoSceneCommand();
    });

    module.def("create_box_collision", [](const std::string& name,
                                          const py::handle& size,
                                          const py::handle& position) {
        EnsureRuntimeContext();
        return MakeCollisionShape3DHandle(CreateBoxCollision(name, PythonToVector3(size), PythonToVector3(position)),
                                          PyNodeOwnership::DetachedOwned);
    }, py::arg("name"), py::arg("size"), py::arg("position") = py::make_tuple(0.0, 0.0, 0.0),
       py::return_value_policy::move);

    module.def("create_box_visual", [](const std::string& name,
                                       const py::handle& size,
                                       const py::handle& position) {
        EnsureRuntimeContext();
        return MakeMeshInstance3DHandle(CreateBoxVisual(name, PythonToVector3(size), PythonToVector3(position)),
                                        PyNodeOwnership::DetachedOwned);
    }, py::arg("name"), py::arg("size"), py::arg("position") = py::make_tuple(0.0, 0.0, 0.0),
       py::return_value_policy::move);

    module.def("save_scene", [](PyNodeHandle& root, const std::string& path) {
        EnsureRuntimeContext();
        if (!SaveSceneRoot(root.Resolve(), path)) {
            throw std::runtime_error("failed to save Gobot scene to '" + path + "'");
        }
    }, py::arg("root"), py::arg("path"));

    module.def("import_mjcf_scene", [](const std::string& xml_path,
                                       const std::string& scene_path,
                                       const std::optional<std::string>& name,
                                       const std::optional<std::string>& script) {
        EnsureRuntimeContext();
        ImportMJCFScene(xml_path, scene_path, name, script);
    }, py::arg("xml_path"), py::arg("scene_path"), py::arg("name") = py::none(), py::arg("script") = py::none());

    module.def("load_resource", [](const std::string& path, const std::string& type_hint) {
        EnsureRuntimeContext();
        return ResourceToPythonDict(ResourceLoader::Load(path, type_hint));
    }, py::arg("path"), py::arg("type_hint") = "");

    module.def("_node_from_id", [](std::uint64_t id, const py::object& context_object) -> py::object {
        EnsureRuntimeContext();
        auto* node = Object::PointerCastTo<Node>(ObjectDB::GetInstance(ObjectID(id)));
        if (node == nullptr) {
            return py::none();
        }
        EngineContext* context = context_object.is_none() ? nullptr : py::cast<EngineContext*>(context_object);
        EngineContext* handle_context = context != nullptr ? context : ContextForNode(node);
        return MakeTypedNodeObject(node,
                                   PyNodeOwnership::Borrowed,
                                   handle_context,
                                   SceneEpochForContext(handle_context));
    }, py::arg("id"), py::arg("context") = py::none());

    module.def("_capture_rgb",
               [](const py::object& root,
                  int width,
                  int height,
                  const py::handle& eye,
                  const py::handle& target,
                  const py::handle& up,
                  RealType fov_y,
                  RealType z_near,
                  RealType z_far,
                  const py::handle& debug_arrows) {
                   EnsureRuntimeContext();
                   return CaptureRgb(root, width, height, eye, target, up, fov_y, z_near, z_far, debug_arrows);
               },
               py::arg("root") = py::none(),
               py::arg("width") = 640,
               py::arg("height") = 480,
               py::arg("eye") = py::make_tuple(2.4, -3.0, 1.6),
               py::arg("target") = py::make_tuple(0.0, 0.0, 0.5),
               py::arg("up") = py::make_tuple(0.0, 0.0, 1.0),
               py::arg("fov_y") = 60.0,
               py::arg("z_near") = 0.05,
               py::arg("z_far") = 200.0,
               py::arg("debug_arrows") = py::none());

    auto set_debug_arrows = [](const py::handle& debug_arrows) {
        EnsureRuntimeContext();
        GetActiveAppContext().SetDebugArrows(PythonToDebugArrows(debug_arrows));
    };
    module.def("_set_debug_arrows", set_debug_arrows, py::arg("debug_arrows"));
    module.def("set_debug_arrows", set_debug_arrows, py::arg("debug_arrows"));

    auto clear_debug_arrows = []() {
        EnsureRuntimeContext();
        GetActiveAppContext().ClearDebugArrows();
    };
    module.def("_clear_debug_arrows", clear_debug_arrows);
    module.def("clear_debug_arrows", clear_debug_arrows);

    module.def("_shutdown_headless_render_context", []() {
        ShutdownHeadlessRenderContext();
    });

    module.def("create_test_scene", []() {
        EnsureRuntimeContext();
        return std::make_unique<PyScene>(CreateTestRobotScene());
    });

    module.def("backend_infos", []() {
        EnsureRuntimeContext();
        py::list infos;
        for (const PhysicsBackendInfo& info : PhysicsServer::GetBackendInfosForAllBackends()) {
            infos.append(ReflectedToPythonDict(info));
        }
        return infos;
    });

    module.def("_task_llvm_available", []() {
        return rl::TaskLlvmAvailable();
    });
    module.def("_task_llvm_version", []() {
        return rl::TaskLlvmVersion();
    });
    module.def("_task_llvm_last_error", []() {
        return rl::TaskLlvmLastError();
    });
    module.def("_task_llvm_compile_cpp",
               [](const std::string& source,
                  const std::string& virtual_path,
                  const std::string& object_path,
                  int optimization_level,
                  bool fast_math,
                  bool debug) {
                   rl::TaskLlvmCompileOptions options;
                   options.optimization_level = optimization_level;
                   options.fast_math = fast_math;
                   options.debug = debug;
                   const int status = rl::TaskLlvmCompileCpp(source, virtual_path, object_path, options);
                   if (status != 0) {
                       throw std::runtime_error("task LLVM compile failed: " + rl::TaskLlvmLastError());
                   }
                   return status;
               },
               py::arg("source"),
               py::arg("virtual_path"),
               py::arg("object_path"),
               py::arg("optimization_level") = 3,
               py::arg("fast_math") = true,
               py::arg("debug") = false);
    module.def("_task_llvm_load_obj", [](const std::string& object_path, const std::string& module_name) {
        const int status = rl::TaskLlvmLoadObject(object_path, module_name);
        if (status != 0) {
            throw std::runtime_error("task LLVM load failed: " + rl::TaskLlvmLastError());
        }
        return status;
    }, py::arg("object_path"), py::arg("module_name"));
    module.def("_task_llvm_lookup", [](const std::string& module_name, const std::string& symbol_name) {
        const std::uint64_t address = rl::TaskLlvmLookup(module_name, symbol_name);
        if (address == 0) {
            throw std::runtime_error("task LLVM lookup failed: " + rl::TaskLlvmLastError());
        }
        return address;
    }, py::arg("module_name"), py::arg("symbol_name"));
    module.def("_task_llvm_unload", [](const std::string& module_name) {
        const int status = rl::TaskLlvmUnload(module_name);
        if (status != 0) {
            throw std::runtime_error("task LLVM unload failed: " + rl::TaskLlvmLastError());
        }
        return status;
    }, py::arg("module_name"));
}

} // namespace gobot::python
