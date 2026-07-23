#include "manual_bindings_internal.hpp"

#include <dlpack.h>

namespace gobot {

class PythonRenderBufferBridge {
public:
    static void* DevicePointer(const RenderBuffer& buffer) {
        return buffer.storage_ != nullptr ? buffer.storage_->device_pointer : nullptr;
    }

    static std::size_t PixelStrideBytes(const RenderBuffer& buffer) {
        return buffer.storage_ != nullptr ? buffer.storage_->pixel_stride_bytes : 0;
    }

    static int DeviceId(const RenderBuffer& buffer) {
        return buffer.storage_ != nullptr ? buffer.storage_->device_id : 0;
    }
};

} // namespace gobot

namespace gobot::python {

namespace {

RenderOutputType ParseRenderOutput(const std::string& name) {
    if (name == "rgb") return RenderOutputType::Rgb;
    if (name == "linear_depth") return RenderOutputType::LinearDepth;
    if (name == "world_normal") return RenderOutputType::WorldNormal;
    if (name == "instance_id") return RenderOutputType::InstanceId;
    if (name == "semantic_id") return RenderOutputType::SemanticId;
    throw std::invalid_argument("unknown render output '" + name + "'");
}

RenderDevice ParseRenderDevice(const std::string& name) {
    if (name == "auto") return RenderDevice::Auto;
    if (name == "cpu") return RenderDevice::Cpu;
    if (name == "cuda") return RenderDevice::Cuda;
    throw std::invalid_argument("unknown render device '" + name + "'");
}

RenderProductMode ParseRenderProductMode(const std::string& name) {
    if (name == "minimal") return RenderProductMode::Minimal;
    if (name == "realtime") return RenderProductMode::Realtime;
    if (name == "progressive") return RenderProductMode::Progressive;
    throw std::invalid_argument("unknown render product mode '" + name + "'");
}

RasterAntiAliasingMode ParseRasterAntiAliasing(const std::string& name) {
    if (name == "disabled") return RasterAntiAliasingMode::Disabled;
    if (name == "fxaa") return RasterAntiAliasingMode::Fxaa;
    throw std::invalid_argument("unknown raster anti-aliasing mode '" + name + "'");
}

RasterShadowQuality ParseRasterShadowQuality(const std::string& name) {
    if (name == "disabled") return RasterShadowQuality::Disabled;
    if (name == "low") return RasterShadowQuality::Low;
    if (name == "medium") return RasterShadowQuality::Medium;
    if (name == "high") return RasterShadowQuality::High;
    throw std::invalid_argument("unknown raster shadow quality '" + name + "'");
}

const char* RasterAntiAliasingName(RasterAntiAliasingMode mode) {
    return mode == RasterAntiAliasingMode::Fxaa ? "fxaa" : "disabled";
}

const char* RasterShadowQualityName(RasterShadowQuality quality) {
    switch (quality) {
        case RasterShadowQuality::Disabled: return "disabled";
        case RasterShadowQuality::Low: return "low";
        case RasterShadowQuality::Medium: return "medium";
        case RasterShadowQuality::High: return "high";
    }
    return "disabled";
}

std::string RenderDeviceName(RenderDevice device) {
    switch (device) {
        case RenderDevice::Auto: return "auto";
        case RenderDevice::Cpu: return "cpu";
        case RenderDevice::Cuda: return "cuda";
    }
    return "unknown";
}

std::string RenderDataTypeName(RenderDataType type) {
    switch (type) {
        case RenderDataType::UInt8: return "uint8";
        case RenderDataType::Float32: return "float32";
        case RenderDataType::UInt32: return "uint32";
    }
    return "unknown";
}

py::array RenderBufferNumpy(const std::shared_ptr<RenderBuffer>& buffer) {
    if (buffer == nullptr) {
        throw std::runtime_error("cannot convert a null render buffer to NumPy");
    }
    py::dtype dtype;
    switch (buffer->GetDataType()) {
        case RenderDataType::UInt8: dtype = py::dtype::of<std::uint8_t>(); break;
        case RenderDataType::Float32: dtype = py::dtype::of<float>(); break;
        case RenderDataType::UInt32: dtype = py::dtype::of<std::uint32_t>(); break;
    }

    const std::size_t item_size = RenderDataTypeSize(buffer->GetDataType());
    const std::size_t channels = buffer->GetChannelCount();
    std::vector<py::ssize_t> shape = {
            static_cast<py::ssize_t>(buffer->GetHeight()),
            static_cast<py::ssize_t>(buffer->GetWidth())};
    std::vector<py::ssize_t> strides = {
            static_cast<py::ssize_t>(buffer->GetWidth() * channels * item_size),
            static_cast<py::ssize_t>(channels * item_size)};
    if (channels != 1) {
        shape.push_back(static_cast<py::ssize_t>(channels));
        strides.push_back(static_cast<py::ssize_t>(item_size));
    }

    py::array array;
    if (buffer->GetDevice() == RenderDevice::Cpu) {
        auto* owner = new std::shared_ptr<RenderBuffer>(buffer);
        py::capsule capsule(owner, [](void* pointer) {
            delete static_cast<std::shared_ptr<RenderBuffer>*>(pointer);
        });
        array = py::array(dtype,
                          std::move(shape),
                          std::move(strides),
                          const_cast<void*>(buffer->GetData()),
                          capsule);
    } else {
        array = py::array(dtype, std::move(shape));
        if (!buffer->CopyToHost(array.mutable_data(), buffer->GetByteSize())) {
            throw std::runtime_error("failed to copy CUDA render buffer to host memory");
        }
    }
    array.attr("setflags")(false);
    return array;
}

void DeleteRenderBufferDlpack(DLManagedTensor* tensor) {
    delete[] tensor->dl_tensor.shape;
    delete[] tensor->dl_tensor.strides;
    delete static_cast<std::shared_ptr<RenderBuffer>*>(tensor->manager_ctx);
    delete tensor;
}

void DeleteRenderBufferDlpackCapsule(PyObject* capsule) {
    if (!PyCapsule_IsValid(capsule, "dltensor")) {
        return;
    }
    auto* tensor = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
    if (tensor != nullptr && tensor->deleter != nullptr) {
        tensor->deleter(tensor);
    }
}

py::capsule RenderBufferCudaDlpack(const std::shared_ptr<RenderBuffer>& buffer) {
    void* device_pointer = PythonRenderBufferBridge::DevicePointer(*buffer);
    if (device_pointer == nullptr) {
        throw std::runtime_error("CUDA render buffer has no device allocation");
    }

    const std::size_t channels = buffer->GetChannelCount();
    const std::size_t item_size = RenderDataTypeSize(buffer->GetDataType());
    std::size_t pixel_stride = PythonRenderBufferBridge::PixelStrideBytes(*buffer);
    if (pixel_stride == 0) {
        pixel_stride = channels * item_size;
    }
    if (pixel_stride % item_size != 0) {
        throw std::runtime_error("CUDA render buffer has an invalid DLPack stride");
    }

    auto* tensor = new DLManagedTensor{};
    tensor->dl_tensor.data = device_pointer;
    tensor->dl_tensor.device = {kDLCUDA, PythonRenderBufferBridge::DeviceId(*buffer)};
    tensor->dl_tensor.ndim = channels == 1 ? 2 : 3;
    switch (buffer->GetDataType()) {
        case RenderDataType::UInt8:
            tensor->dl_tensor.dtype = {kDLUInt, 8, 1};
            break;
        case RenderDataType::Float32:
            tensor->dl_tensor.dtype = {kDLFloat, 32, 1};
            break;
        case RenderDataType::UInt32:
            tensor->dl_tensor.dtype = {kDLUInt, 32, 1};
            break;
    }
    tensor->dl_tensor.shape = new std::int64_t[tensor->dl_tensor.ndim];
    tensor->dl_tensor.strides = new std::int64_t[tensor->dl_tensor.ndim];
    tensor->dl_tensor.shape[0] = buffer->GetHeight();
    tensor->dl_tensor.shape[1] = buffer->GetWidth();
    const std::int64_t pixel_stride_elements = static_cast<std::int64_t>(pixel_stride / item_size);
    tensor->dl_tensor.strides[0] = buffer->GetWidth() * pixel_stride_elements;
    tensor->dl_tensor.strides[1] = pixel_stride_elements;
    if (channels != 1) {
        tensor->dl_tensor.shape[2] = static_cast<std::int64_t>(channels);
        tensor->dl_tensor.strides[2] = 1;
    }
    tensor->dl_tensor.byte_offset = 0;
    tensor->manager_ctx = new std::shared_ptr<RenderBuffer>(buffer);
    tensor->deleter = &DeleteRenderBufferDlpack;
    return py::capsule(tensor, "dltensor", &DeleteRenderBufferDlpackCapsule);
}

py::dict IdMapToPython(const std::map<std::uint32_t, std::string>& values) {
    py::dict result;
    for (const auto& [id, value] : values) {
        result[py::int_(id)] = value;
    }
    return result;
}

void EnsureRenderContext() {
    if (RenderServer::HasInstance()) {
        return;
    }
    HeadlessRenderContext& context = EnsureHeadlessRenderContext();
    if (!context.Initialize()) {
        throw std::runtime_error(context.GetLastError());
    }
}

} // namespace

void RegisterManualModuleFunctions(py::module_& module) {
    py::class_<RenderBuffer, std::shared_ptr<RenderBuffer>>(module, "RenderBuffer")
            .def_property_readonly("name", [](const RenderBuffer& buffer) {
                return std::string(RenderOutputTypeName(buffer.GetOutputType()));
            })
            .def_property_readonly("dtype", [](const RenderBuffer& buffer) {
                return RenderDataTypeName(buffer.GetDataType());
            })
            .def_property_readonly("device", [](const RenderBuffer& buffer) {
                return RenderDeviceName(buffer.GetDevice());
            })
            .def_property_readonly("shape", [](const RenderBuffer& buffer) {
                return buffer.GetShape();
            })
            .def_property_readonly("nbytes", &RenderBuffer::GetByteSize)
            .def("numpy", &RenderBufferNumpy)
            .def("__dlpack_device__", [](const RenderBuffer& buffer) {
                const int device_id = buffer.GetDevice() == RenderDevice::Cuda
                                              ? PythonRenderBufferBridge::DeviceId(buffer)
                                              : 0;
                return py::make_tuple(buffer.GetDevice() == RenderDevice::Cuda ? 2 : 1,
                                      device_id);
            })
            .def("__dlpack__", [](const std::shared_ptr<RenderBuffer>& buffer,
                                   py::args args,
                                   py::kwargs kwargs) {
                if (buffer->GetDevice() == RenderDevice::Cuda) {
                    return py::object(RenderBufferCudaDlpack(buffer));
                }
                py::array array = RenderBufferNumpy(buffer);
                return array.attr("__dlpack__")(*args, **kwargs);
            });

    py::class_<RenderFrame, std::shared_ptr<RenderFrame>>(module, "RenderFrame")
            .def_property_readonly("frame_index", &RenderFrame::GetFrameIndex)
            .def_property_readonly("instance_id_to_path", [](const RenderFrame& frame) {
                return IdMapToPython(frame.GetInstancePaths());
            })
            .def_property_readonly("semantic_id_to_label", [](const RenderFrame& frame) {
                return IdMapToPython(frame.GetSemanticLabels());
            })
            .def("keys", [](const RenderFrame& frame) {
                py::list result;
                for (const RenderOutputType output : frame.GetOutputs()) {
                    result.append(RenderOutputTypeName(output));
                }
                return result;
            })
            .def("__len__", [](const RenderFrame& frame) { return frame.GetOutputs().size(); })
            .def("__contains__", [](const RenderFrame& frame, const std::string& name) {
                try {
                    return frame.Contains(ParseRenderOutput(name));
                } catch (const std::invalid_argument&) {
                    return false;
                }
            })
            .def("__getitem__", [](const RenderFrame& frame, const std::string& name) {
                std::shared_ptr<RenderBuffer> buffer = frame.Get(ParseRenderOutput(name));
                if (buffer == nullptr) {
                    throw py::key_error(name);
                }
                return buffer;
            });

    py::class_<RenderProduct, std::shared_ptr<RenderProduct>>(module, "RenderProduct")
            .def_property_readonly("width", [](const RenderProduct& product) {
                return product.GetDesc().width;
            })
            .def_property_readonly("height", [](const RenderProduct& product) {
                return product.GetDesc().height;
            })
            .def_property_readonly("device", [](const RenderProduct& product) {
                return RenderDeviceName(product.GetDevice());
            });

    py::class_<CameraSensor, std::shared_ptr<CameraSensor>>(module, "_CameraSensor")
            .def(py::init([](const PyNodeHandle& camera_handle,
                             const py::object& root_object,
                             int width,
                             int height,
                             const std::vector<std::string>& output_names,
                             const std::string& device,
                             const std::string& mode,
                             std::size_t frame_slots) {
                EnsureRuntimeContext();
                auto* camera = Object::PointerCastTo<Camera3D>(camera_handle.Resolve());
                if (camera == nullptr) {
                    throw py::type_error("CameraSensor camera must reference a Camera3D node");
                }
                Node* root = root_object.is_none()
                                     ? ActiveSceneRoot()
                                     : py::cast<const PyNodeHandle&>(root_object).Resolve();
                if (root == nullptr) {
                    throw std::runtime_error("CameraSensor requires a scene root");
                }
                RenderProductDesc desc;
                desc.width = width;
                desc.height = height;
                desc.device = ParseRenderDevice(device);
                desc.mode = ParseRenderProductMode(mode);
                desc.frame_slots = frame_slots;
                desc.outputs.clear();
                for (const std::string& output_name : output_names) {
                    desc.outputs.push_back(ParseRenderOutput(output_name));
                }
                return std::make_shared<CameraSensor>(camera, root, std::move(desc));
            }),
            py::arg("camera"),
            py::arg("root") = py::none(),
            py::arg("width") = 640,
            py::arg("height") = 480,
            py::arg("outputs") = std::vector<std::string>{"rgb"},
            py::arg("device") = "auto",
            py::arg("mode") = "minimal",
            py::arg("frame_slots") = 3)
            .def_property_readonly("render_product", &CameraSensor::GetRenderProduct)
            .def("capture", [](CameraSensor& sensor) {
                EnsureRenderContext();
                return sensor.Capture();
            });

    module.def("_get_raster_settings", []() {
        EnsureRenderContext();
        const RasterRendererSettings settings =
                RenderServer::GetInstance()->GetSceneRendererSettings().raster;
        py::dict result;
        result["frustum_culling"] = settings.frustum_culling;
        result["anti_aliasing"] = RasterAntiAliasingName(settings.anti_aliasing);
        result["shadow_quality"] = RasterShadowQualityName(settings.shadow_quality);
        result["shadow_distance"] = settings.shadow_distance;
        return result;
    });
    module.def("_set_raster_settings", [](const py::dict& values) {
        EnsureRenderContext();
        SceneRendererSettings settings = RenderServer::GetInstance()->GetSceneRendererSettings();
        for (const auto& entry : values) {
            const std::string key = py::cast<std::string>(entry.first);
            if (key == "frustum_culling") {
                settings.raster.frustum_culling = py::cast<bool>(entry.second);
            } else if (key == "anti_aliasing") {
                settings.raster.anti_aliasing =
                        ParseRasterAntiAliasing(py::cast<std::string>(entry.second));
            } else if (key == "shadow_quality") {
                settings.raster.shadow_quality =
                        ParseRasterShadowQuality(py::cast<std::string>(entry.second));
            } else if (key == "shadow_distance") {
                settings.raster.shadow_distance = py::cast<RealType>(entry.second);
                if (!std::isfinite(settings.raster.shadow_distance) ||
                    settings.raster.shadow_distance <= 0.0) {
                    throw py::value_error("shadow_distance must be a finite positive number");
                }
            } else {
                throw py::key_error("unknown raster setting '" + key + "'");
            }
        }
        RenderServer::GetInstance()->SetSceneRendererSettings(settings);
    });

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

}

} // namespace gobot::python
