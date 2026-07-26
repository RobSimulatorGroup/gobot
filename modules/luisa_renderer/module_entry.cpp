#include "luisa_renderer_internal.hpp"

#include <exception>

namespace gobot::luisa_renderer {
namespace {

void* CreateRenderer(const char* module_directory, char* error, std::size_t error_size) {
    try {
        return new LuisaRenderer(module_directory != nullptr ? module_directory : ".");
    } catch (const std::exception& exception) {
        SetError(error, error_size, exception.what());
        return nullptr;
    }
}

void DestroyRenderer(void* renderer) {
    delete static_cast<LuisaRenderer*>(renderer);
}

gobot::SceneRendererCapabilities GetCapabilities(void* renderer) {
    return renderer != nullptr ? static_cast<LuisaRenderer*>(renderer)->Capabilities()
                               : MakeCapabilities();
}

gobot::LuisaRendererResult Render(void* renderer,
                                  const gobot::LuisaRendererTarget* target,
                                  const gobot::RenderSceneSnapshot* snapshot,
                                  const gobot::RenderViewSnapshot* view,
                                  const gobot::SceneRendererSettings* settings,
                                  gobot::SceneRendererStats* stats,
                                  char* error,
                                  std::size_t error_size) {
    if (renderer == nullptr || target == nullptr || snapshot == nullptr || view == nullptr || settings == nullptr) {
        SetError(error, error_size, "Invalid Luisa renderer call arguments.");
        return gobot::LuisaRendererResult::FatalError;
    }
    try {
        std::string message;
        const auto result = static_cast<LuisaRenderer*>(renderer)->Render(
                *target, *snapshot, *view, *settings, stats, &message);
        if (result != gobot::LuisaRendererResult::Success) {
            SetError(error, error_size, message);
        }
        return result;
    } catch (const std::exception& exception) {
        SetError(error, error_size, exception.what());
        return gobot::LuisaRendererResult::FatalError;
    }
}

gobot::LuisaRendererResult RenderGaussianBackground(
        void* renderer,
        const gobot::LuisaRendererTarget* target,
        const gobot::RenderSceneSnapshot* snapshot,
        const gobot::RenderViewSnapshot* view,
        gobot::SceneRendererStats* stats,
        char* error,
        std::size_t error_size) {
    if (renderer == nullptr || target == nullptr || snapshot == nullptr || view == nullptr) {
        SetError(error, error_size, "Invalid Gaussian background render call arguments.");
        return gobot::LuisaRendererResult::FatalError;
    }
    try {
        std::string message;
        const gobot::LuisaRendererResult result =
                static_cast<LuisaRenderer*>(renderer)->RenderGaussianBackground(
                        *target, *snapshot, *view, stats, &message);
        if (result != gobot::LuisaRendererResult::Success) {
            SetError(error, error_size, message);
        }
        return result;
    } catch (const std::exception& exception) {
        SetError(error, error_size, exception.what());
        return gobot::LuisaRendererResult::FatalError;
    }
}

void ResetAccumulation(void* renderer) {
    if (renderer != nullptr) {
        static_cast<LuisaRenderer*>(renderer)->ResetAccumulation();
    }
}

gobot::LuisaRendererResult CaptureRenderProduct(
        void* renderer,
        const gobot::RenderSceneSnapshot* snapshot,
        const gobot::RenderViewSnapshot* view,
        const gobot::LuisaRenderProductRequest* request,
        gobot::LuisaRenderProductFrame* frame,
        char* error,
        std::size_t error_size) {
    if (renderer == nullptr || snapshot == nullptr || view == nullptr || request == nullptr ||
        frame == nullptr) {
        SetError(error, error_size, "Invalid Luisa CUDA render-product call arguments.");
        return gobot::LuisaRendererResult::FatalError;
    }
    try {
        std::string message;
        const gobot::LuisaRendererResult result =
                static_cast<LuisaRenderer*>(renderer)->CaptureRenderProduct(
                        *snapshot, *view, *request, frame, &message);
        if (result != gobot::LuisaRendererResult::Success) {
            SetError(error, error_size, message);
        }
        return result;
    } catch (const std::exception& exception) {
        SetError(error, error_size, exception.what());
        return gobot::LuisaRendererResult::FatalError;
    }
}

void ReleaseRenderProduct(void*, void* frame) {
    delete static_cast<CudaRenderProductFrame*>(frame);
}

bool ReadbackRenderProduct(void* renderer,
                           void* frame,
                           std::uint32_t output,
                           void* destination,
                           std::size_t destination_size,
                           char* error,
                           std::size_t error_size) {
    if (renderer == nullptr || frame == nullptr) {
        SetError(error, error_size, "Invalid Luisa CUDA render-product readback arguments.");
        return false;
    }
    try {
        std::string message;
        const bool success = static_cast<LuisaRenderer*>(renderer)->ReadbackRenderProduct(
                static_cast<CudaRenderProductFrame*>(frame),
                output,
                destination,
                destination_size,
                &message);
        if (!success) {
            SetError(error, error_size, message);
        }
        return success;
    } catch (const std::exception& exception) {
        SetError(error, error_size, exception.what());
        return false;
    }
}

const gobot::LuisaRendererModuleApi kApi{
        gobot::GOBOT_LUISA_RENDERER_ABI_VERSION,
        &CreateRenderer,
        &DestroyRenderer,
        &GetCapabilities,
        &Render,
        &RenderGaussianBackground,
        &ResetAccumulation,
        &CaptureRenderProduct,
        &ReleaseRenderProduct,
        &ReadbackRenderProduct};

} // namespace

extern "C" __attribute__((visibility("default")))
const gobot::LuisaRendererModuleApi* gobot_luisa_renderer_get_api() {
    return &kApi;
}

} // namespace gobot::luisa_renderer
