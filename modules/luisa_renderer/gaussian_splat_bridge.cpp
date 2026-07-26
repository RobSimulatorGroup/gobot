#include "luisa_renderer_internal.hpp"

#if defined(GOBOT_HAS_GSPLAT_INFERENCE)
#include <gsplat_inference/renderer.h>
#endif

#include <cmath>

namespace gobot::luisa_renderer {

#if defined(GOBOT_HAS_GSPLAT_INFERENCE)
void GaussianRendererDeleter::operator()(gsplat_inference::Renderer* renderer) const noexcept {
    if (renderer == nullptr) {
        return;
    }
    CudaContextScope context_scope{context};
    if (context_scope) {
        gsplat_inference::Destroy(renderer);
    }
}
#endif

bool LuisaRenderer::EnqueueGaussianToBuffer(const gobot::RenderSceneSnapshot& snapshot,
                                            const gobot::RenderViewSnapshot& view,
                                            int width,
                                            int height,
                                            void* output,
                                            const void* geometry_coverage,
                                            bool top_left_origin,
                                            std::string* error) {
    if (!snapshot.gaussian_splat_error.empty()) {
        *error = snapshot.gaussian_splat_error;
        return false;
    }
    if (snapshot.gaussian_splats.size() != 1u || output == nullptr) {
        *error = "Gaussian rendering requires exactly one valid GaussianSplat3D and an output buffer.";
        return false;
    }
#if !defined(GOBOT_HAS_GSPLAT_INFERENCE)
    (void)view;
    (void)width;
    (void)height;
    (void)geometry_coverage;
    (void)top_left_origin;
    *error = "This Luisa renderer was built without gsplat CUDA inference support.";
    return false;
#else
    const gobot::GaussianSplatRenderItem& item = snapshot.gaussian_splats.front();
    if (!item.IsValid() || gaussian_renderer_ == nullptr) {
        *error = "Render snapshot contains invalid Gaussian Splatting data.";
        return false;
    }
    const gobot::Matrix3 linear = item.model.template block<3, 3>(0, 0);
    const gobot::Vector3 scales{
            linear.col(0).norm(), linear.col(1).norm(), linear.col(2).norm()};
    const float mean_scale = static_cast<float>(scales.mean());
    if (!(mean_scale > 0.0f) ||
        (scales.array() - scales.mean()).abs().maxCoeff() > 1e-5 * scales.mean()) {
        *error = "GaussianSplat3D requires a positive uniform world scale.";
        return false;
    }
    const gobot::Matrix3 rotation = linear / scales.mean();
    if (!(rotation.transpose() * rotation).isApprox(gobot::Matrix3::Identity(), 1e-5) ||
        std::abs(rotation.determinant() - 1.0) > 1e-5) {
        *error = "GaussianSplat3D world transform must not contain shear or reflection.";
        return false;
    }

    CudaContextScope context_scope{NativeContext()};
    if (!context_scope) {
        *error = CudaError(context_scope.GetResult(), "cuCtxPushCurrent for gsplat");
        return false;
    }
    CUstream stream = NativeStream();

    bool success = true;
    std::array<char, 1024> gsplat_error{};
    const std::uint64_t resource_id = item.resource_id.operator std::uint64_t();
    if (gaussian_resource_id_ != resource_id ||
        gaussian_resource_revision_ != item.resource_revision) {
        const gobot::GaussianSplatData& data = *item.data;
        const gsplat_inference::SceneData scene_data{
                data.means.data(),
                data.rotations_wxyz.data(),
                data.scales.data(),
                data.opacities.data(),
                data.sh_coefficients.data(),
                data.count,
                data.sh_degree};
        success = gsplat_inference::Upload(gaussian_renderer_.get(),
                                           scene_data,
                                           reinterpret_cast<void*>(stream),
                                           gsplat_error.data(),
                                           gsplat_error.size());
        if (success) {
            gaussian_resource_id_ = resource_id;
            gaussian_resource_revision_ = item.resource_revision;
        }
    }

    if (success) {
        const std::array<float, 16> view_matrix = ToFloatMatrix(view.camera.view);
        const std::array<float, 16> projection_matrix = ToFloatMatrix(view.camera.projection);
        const std::array<float, 16> model_matrix = ToFloatMatrix(item.model);
        gsplat_inference::CameraData camera;
        camera.view_column_major = view_matrix.data();
        camera.projection_column_major = projection_matrix.data();
        camera.model_column_major = model_matrix.data();
        camera.camera_position[0] = static_cast<float>(view.camera.world_position.x());
        camera.camera_position[1] = static_cast<float>(view.camera.world_position.y());
        camera.camera_position[2] = static_cast<float>(view.camera.world_position.z());
        camera.clear_color[0] = snapshot.environment.clear_color.red();
        camera.clear_color[1] = snapshot.environment.clear_color.green();
        camera.clear_color[2] = snapshot.environment.clear_color.blue();
        camera.near_plane = static_cast<float>(view.camera.z_near);
        camera.far_plane = static_cast<float>(view.camera.z_far);
        camera.width = width;
        camera.height = height;
        camera.top_left_origin = top_left_origin;
        const gsplat_inference::RenderTarget target{
                static_cast<std::uint32_t*>(output),
                static_cast<const std::uint32_t*>(geometry_coverage)};
        success = gsplat_inference::Render(gaussian_renderer_.get(),
                                           camera,
                                           target,
                                           reinterpret_cast<void*>(stream),
                                           gsplat_error.data(),
                                           gsplat_error.size());
    }
    if (!success && error->empty()) {
        *error = gsplat_error[0] != '\0' ? gsplat_error.data() : "gsplat inference failed.";
    }
    return success;
#endif
}

gobot::LuisaRendererResult LuisaRenderer::RenderGaussianBackground(
        const gobot::LuisaRendererTarget& target,
        const gobot::RenderSceneSnapshot& snapshot,
        const gobot::RenderViewSnapshot& view,
        gobot::SceneRendererStats* stats,
        std::string* error) {
    if (snapshot.gaussian_splats.empty()) return gobot::LuisaRendererResult::Success;
    if (target.gl_color_texture == 0u || !EnsureFrameResources(target.width, target.height, error)) {
        if (error->empty()) *error = "Gaussian raster presentation requires a valid OpenGL color texture.";
        return gobot::LuisaRendererResult::RecoverableError;
    }
    if (!gaussian_start_event_.Record(NativeStream(), error) ||
        !EnqueueGaussianToBuffer(snapshot,
                                 view,
                                 target.width,
                                 target.height,
                                 presentation_.native_handle(),
                                 nullptr,
                                 false,
                                 error) ||
        !gaussian_end_event_.Record(NativeStream(), error)) {
        return gobot::LuisaRendererResult::RecoverableError;
    }
    if (!Present(target, error)) {
        return gobot::LuisaRendererResult::RecoverableError;
    }
    if (stats != nullptr) {
        if (!gaussian_start_event_.ElapsedMilliseconds(
                    gaussian_end_event_, &stats->gaussian_splat_ms, error)) {
            return gobot::LuisaRendererResult::RecoverableError;
        }
        stats->gaussian_count = snapshot.gaussian_splats.front().data->count;
    }
    return gobot::LuisaRendererResult::Success;
}

} // namespace gobot::luisa_renderer
