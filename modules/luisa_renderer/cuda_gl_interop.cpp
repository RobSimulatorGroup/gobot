#include "luisa_renderer_internal.hpp"

namespace gobot::luisa_renderer {

CudaContextScope::CudaContextScope(CUcontext context) noexcept {
    if (context != nullptr) {
        result_ = cuCtxPushCurrent(context);
        active_ = result_ == CUDA_SUCCESS;
    }
}

CudaContextScope::~CudaContextScope() noexcept {
    if (active_) {
        CUcontext popped = nullptr;
        cuCtxPopCurrent(&popped);
    }
}

CudaGraphicsImage::~CudaGraphicsImage() noexcept {
    try {
        (void)Reset();
    } catch (...) {
        // CUDA teardown is best-effort and must not escape a destructor.
    }
}

bool CudaGraphicsImage::Ensure(std::uint32_t texture, std::string* error) {
    if (resource_ != nullptr && texture_ == texture) {
        return true;
    }
    if (!Reset(error)) {
        return false;
    }
    if (context_ == nullptr || texture == 0u) {
        *error = "Invalid CUDA-OpenGL texture registration request.";
        return false;
    }
    CudaContextScope context_scope{context_};
    if (!context_scope) {
        *error = CudaError(context_scope.GetResult(), "cuCtxPushCurrent");
        return false;
    }
    const CUresult result = cuGraphicsGLRegisterImage(&resource_,
                                                      texture,
                                                      GL_TEXTURE_2D,
                                                      CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
    if (result != CUDA_SUCCESS) {
        resource_ = nullptr;
        *error = CudaError(result, "cuGraphicsGLRegisterImage");
        return false;
    }
    texture_ = texture;
    return true;
}

bool CudaGraphicsImage::Reset(std::string* error) {
    if (resource_ == nullptr) {
        texture_ = 0u;
        return true;
    }
    CudaContextScope context_scope{context_};
    if (!context_scope) {
        if (error != nullptr) {
            *error = CudaError(context_scope.GetResult(),
                               "cuCtxPushCurrent for texture unregistration");
        }
        return false;
    }
    const CUresult result = cuGraphicsUnregisterResource(resource_);
    if (result != CUDA_SUCCESS) {
        if (error != nullptr) {
            *error = CudaError(result, "cuGraphicsUnregisterResource");
        }
        return false;
    }
    resource_ = nullptr;
    texture_ = 0u;
    return true;
}

CudaGraphicsMap::CudaGraphicsMap(CUgraphicsResource resource, CUstream stream) noexcept
    : resource_(resource), stream_(stream) {
    if (resource_ != nullptr) {
        map_result_ = cuGraphicsMapResources(1, &resource_, stream_);
        mapped_ = map_result_ == CUDA_SUCCESS;
    }
}

CudaGraphicsMap::~CudaGraphicsMap() noexcept {
    if (mapped_) {
        cuGraphicsUnmapResources(1, &resource_, stream_);
    }
}

bool CudaGraphicsMap::GetArray(CUarray* destination, std::string* error) {
    if (!mapped_) {
        *error = CudaError(map_result_, "cuGraphicsMapResources");
        return false;
    }
    const CUresult result =
            cuGraphicsSubResourceGetMappedArray(destination, resource_, 0, 0);
    if (result != CUDA_SUCCESS) {
        *error = CudaError(result, "cuGraphicsSubResourceGetMappedArray");
        return false;
    }
    return true;
}

bool CudaGraphicsMap::Unmap(std::string* error) {
    if (!mapped_) {
        return true;
    }
    const CUresult result = cuGraphicsUnmapResources(1, &resource_, stream_);
    if (result != CUDA_SUCCESS) {
        *error = CudaError(result, "cuGraphicsUnmapResources");
        return false;
    }
    mapped_ = false;
    return true;
}

CudaEvent::~CudaEvent() noexcept {
    if (event_ != nullptr && context_ != nullptr) {
        CudaContextScope context_scope{context_};
        if (context_scope) {
            cuEventDestroy(event_);
        }
    }
}

bool CudaEvent::Ensure(std::string* error) {
    if (event_ != nullptr) {
        return true;
    }
    CudaContextScope context_scope{context_};
    if (!context_scope) {
        *error = CudaError(context_scope.GetResult(), "cuCtxPushCurrent for CUDA event");
        return false;
    }
    const CUresult result = cuEventCreate(&event_, CU_EVENT_DEFAULT);
    if (result != CUDA_SUCCESS) {
        event_ = nullptr;
        *error = CudaError(result, "cuEventCreate");
        return false;
    }
    return true;
}

bool CudaEvent::Record(CUstream stream, std::string* error) {
    if (!Ensure(error)) {
        return false;
    }
    CudaContextScope context_scope{context_};
    if (!context_scope) {
        *error = CudaError(context_scope.GetResult(), "cuCtxPushCurrent for CUDA event");
        return false;
    }
    const CUresult result = cuEventRecord(event_, stream);
    if (result != CUDA_SUCCESS) {
        *error = CudaError(result, "cuEventRecord");
        return false;
    }
    return true;
}

bool CudaEvent::ElapsedMilliseconds(const CudaEvent& end,
                                    double* milliseconds,
                                    std::string* error) const {
    if (event_ == nullptr || end.event_ == nullptr || milliseconds == nullptr) {
        *error = "CUDA event timing requested before both events were recorded.";
        return false;
    }
    CudaContextScope context_scope{context_};
    if (!context_scope) {
        *error = CudaError(context_scope.GetResult(), "cuCtxPushCurrent for CUDA timing");
        return false;
    }
    float elapsed = 0.0f;
    const CUresult result = cuEventElapsedTime(&elapsed, event_, end.event_);
    if (result != CUDA_SUCCESS) {
        *error = CudaError(result, "cuEventElapsedTime");
        return false;
    }
    *milliseconds = static_cast<double>(elapsed);
    return true;
}

CUcontext LuisaRenderer::NativeContext() const noexcept {
    return device_ != nullptr ? reinterpret_cast<CUcontext>(device_->native_handle()) : nullptr;
}

CUstream LuisaRenderer::NativeStream() const noexcept {
    return stream_ != nullptr ? reinterpret_cast<CUstream>(stream_->native_handle()) : nullptr;
}

bool LuisaRenderer::SynchronizeStream(const char* operation, std::string* error) {
    CudaContextScope context_scope{NativeContext()};
    if (!context_scope) {
        *error = CudaError(context_scope.GetResult(), operation);
        return false;
    }
    const CUresult result = cuStreamSynchronize(NativeStream());
    if (result != CUDA_SUCCESS) {
        *error = CudaError(result, operation);
        return false;
    }
    return true;
}

bool LuisaRenderer::EnsureFrameResources(int width, int height, std::string* error) {
    if (width <= 0 || height <= 0) {
        *error = "Invalid Luisa renderer output size.";
        return false;
    }
    if (frame_width_ == width && frame_height_ == height && presentation_) {
        return true;
    }
    if (!interop_.Reset(error)) {
        return false;
    }
    frame_width_ = width;
    frame_height_ = height;
    const uint2 resolution = make_uint2(static_cast<uint>(width), static_cast<uint>(height));
    accumulation_ = device_->create_image<float>(PixelStorage::FLOAT4, resolution);
    guide_ = device_->create_image<float>(PixelStorage::FLOAT4, resolution);
    seeds_ = device_->create_image<uint>(PixelStorage::INT1, resolution);
    presentation_ = device_->create_buffer<uint>(static_cast<std::size_t>(width) * height);
    ResetAccumulation();
    return true;
}

bool LuisaRenderer::Present(const gobot::LuisaRendererTarget& target, std::string* error) {
    if (!interop_.Ensure(target.gl_color_texture, error)) {
        return false;
    }

    CudaContextScope context_scope{NativeContext()};
    if (!context_scope) {
        *error = CudaError(context_scope.GetResult(), "cuCtxPushCurrent for presentation");
        return false;
    }

    CudaGraphicsMap mapping{interop_.Get(), NativeStream()};
    CUarray destination = nullptr;
    if (!mapping.GetArray(&destination, error)) {
        return false;
    }

    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = reinterpret_cast<CUdeviceptr>(presentation_.native_handle());
    copy.srcPitch = static_cast<std::size_t>(target.width) * sizeof(std::uint32_t);
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = destination;
    copy.WidthInBytes = static_cast<std::size_t>(target.width) * sizeof(std::uint32_t);
    copy.Height = target.height;
    const CUresult copy_result = cuMemcpy2DAsync(&copy, NativeStream());
    if (copy_result != CUDA_SUCCESS) {
        *error = CudaError(copy_result, "cuMemcpy2DAsync for presentation");
        return false;
    }
    if (!mapping.Unmap(error)) {
        return false;
    }
    if (!presentation_end_event_.Record(NativeStream(), error)) {
        return false;
    }
    return SynchronizeStream("CUDA-OpenGL presentation", error);
}

} // namespace gobot::luisa_renderer
