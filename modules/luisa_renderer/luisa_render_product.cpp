#include "luisa_renderer_internal.hpp"

namespace gobot::luisa_renderer {

gobot::LuisaRendererResult LuisaRenderer::CaptureRenderProduct(
        const gobot::RenderSceneSnapshot& snapshot,
        const gobot::RenderViewSnapshot& view,
        const gobot::LuisaRenderProductRequest& request,
        gobot::LuisaRenderProductFrame* result,
        std::string* error) {
    if (result == nullptr || request.width <= 0 || request.height <= 0 ||
        request.output_mask == 0u || (request.output_mask & ~kAllProductOutputs) != 0u) {
        *error = "Invalid Luisa CUDA render-product request.";
        return gobot::LuisaRendererResult::RecoverableError;
    }
    if (!SyncScene(snapshot, error, true)) {
        return gobot::LuisaRendererResult::RecoverableError;
    }

    auto frame = std::make_unique<CudaRenderProductFrame>();
    frame->width = request.width;
    frame->height = request.height;
    frame->output_mask = request.output_mask;
    const std::size_t pixel_count = static_cast<std::size_t>(request.width) * request.height;
    if ((request.output_mask & kRgbOutput) != 0u) {
        frame->rgb = device_->create_buffer<uint>(pixel_count);
    }
    if ((request.output_mask & kDepthOutput) != 0u) {
        frame->linear_depth = device_->create_buffer<float>(pixel_count);
    }
    if ((request.output_mask & kNormalOutput) != 0u) {
        frame->world_normal = device_->create_buffer<float4>(pixel_count);
    }
    if ((request.output_mask & kInstanceOutput) != 0u) {
        frame->instance_id = device_->create_buffer<uint>(pixel_count);
    }
    if ((request.output_mask & kSemanticOutput) != 0u) {
        frame->semantic_id = device_->create_buffer<uint>(pixel_count);
    }
    const bool gaussian_rgb_requested = !snapshot.gaussian_splats.empty() &&
                                        (request.output_mask & kRgbOutput) != 0u;
    if (gaussian_rgb_requested) {
        frame->geometry_coverage = device_->create_buffer<uint>(pixel_count);
    }

    const gobot::Matrix4 inverse_view_projection = view.camera.view_projection.inverse();
    const uint2 resolution = make_uint2(static_cast<uint>(request.width),
                                        static_cast<uint>(request.height));
    *stream_ << product_shader_(frame->rgb ? frame->rgb : dummy_uint_,
                                frame->linear_depth ? frame->linear_depth : dummy_float_,
                                frame->world_normal ? frame->world_normal : dummy_float4_,
                                frame->instance_id ? frame->instance_id : dummy_uint_,
                                frame->semantic_id ? frame->semantic_id : dummy_uint_,
                                frame->geometry_coverage ? frame->geometry_coverage : dummy_uint_,
                                accel_,
                                geometry_heap_,
                                texture_heap_,
                                materials_,
                                lights_,
                                instance_ids_,
                                semantic_ids_,
                                rgb_modes_,
                                static_cast<uint>(active_light_count_),
                                environment_texture_slot_,
                                request.output_mask,
                                gaussian_rgb_requested ? 1u : 0u,
                                make_float3(snapshot.environment.sky_color.red(),
                                            snapshot.environment.sky_color.green(),
                                            snapshot.environment.sky_color.blue()),
                                make_float3(snapshot.environment.ground_color.red(),
                                            snapshot.environment.ground_color.green(),
                                            snapshot.environment.ground_color.blue()),
                                static_cast<float>(snapshot.environment.ambient_intensity),
                                static_cast<float>(snapshot.environment.environment_intensity),
                                static_cast<float>(snapshot.environment.exposure),
                                ToLuisaMatrix(inverse_view_projection),
                                ToLuisaMatrix(view.camera.view),
                                ToFloat3(view.camera.world_position))
                            .dispatch(resolution);

    if (gaussian_rgb_requested) {
        if (!EnqueueGaussianToBuffer(snapshot,
                                     view,
                                     request.width,
                                     request.height,
                                     frame->rgb.native_handle(),
                                     frame->geometry_coverage.native_handle(),
                                     true,
                                     error)) {
            return gobot::LuisaRendererResult::RecoverableError;
        }
    }
    if (!SynchronizeStream("CUDA render-product capture", error)) {
        return gobot::LuisaRendererResult::RecoverableError;
    }

    result->frame = frame.get();
    result->device_id = 0;
    if (frame->rgb) {
        result->buffers[0] = {frame->rgb.native_handle(), frame->rgb.size_bytes(), sizeof(uint)};
    }
    if (frame->linear_depth) {
        result->buffers[1] = {
                frame->linear_depth.native_handle(), frame->linear_depth.size_bytes(), sizeof(float)};
    }
    if (frame->world_normal) {
        result->buffers[2] = {frame->world_normal.native_handle(),
                              frame->world_normal.size_bytes(),
                              sizeof(float4)};
    }
    if (frame->instance_id) {
        result->buffers[3] = {
                frame->instance_id.native_handle(), frame->instance_id.size_bytes(), sizeof(uint)};
    }
    if (frame->semantic_id) {
        result->buffers[4] = {
                frame->semantic_id.native_handle(), frame->semantic_id.size_bytes(), sizeof(uint)};
    }
    frame.release();
    return gobot::LuisaRendererResult::Success;
}

bool LuisaRenderer::ReadbackRenderProduct(CudaRenderProductFrame* frame,
                                          std::uint32_t output,
                                          void* destination,
                                          std::size_t destination_size,
                                          std::string* error) {
    if (frame == nullptr || destination == nullptr || output >= 5u ||
        (frame->output_mask & (1u << output)) == 0u) {
        *error = "Invalid Luisa CUDA render-product readback request.";
        return false;
    }
    const std::size_t pixel_count = static_cast<std::size_t>(frame->width) * frame->height;
    switch (output) {
        case 0u: {
            if (destination_size != pixel_count * 3u) {
                *error = "RGB render-product readback size does not match the frame.";
                return false;
            }
            std::vector<uint> rgba(pixel_count);
            *stream_ << frame->rgb.copy_to(luisa::span{rgba}) << synchronize();
            auto* bytes = static_cast<std::uint8_t*>(destination);
            for (std::size_t index = 0; index < pixel_count; ++index) {
                bytes[index * 3u] = static_cast<std::uint8_t>(rgba[index] & 0xffu);
                bytes[index * 3u + 1u] = static_cast<std::uint8_t>((rgba[index] >> 8u) & 0xffu);
                bytes[index * 3u + 2u] = static_cast<std::uint8_t>((rgba[index] >> 16u) & 0xffu);
            }
            return true;
        }
        case 1u:
            if (destination_size != pixel_count * sizeof(float)) {
                *error = "Depth render-product readback size does not match the frame.";
                return false;
            }
            *stream_ << frame->linear_depth.copy_to(
                                luisa::span{static_cast<float*>(destination), pixel_count})
                     << synchronize();
            return true;
        case 2u: {
            if (destination_size != pixel_count * sizeof(float) * 3u) {
                *error = "Normal render-product readback size does not match the frame.";
                return false;
            }
            std::vector<float4> rgba_normal(pixel_count);
            *stream_ << frame->world_normal.copy_to(luisa::span{rgba_normal}) << synchronize();
            auto* values = static_cast<float*>(destination);
            for (std::size_t index = 0; index < pixel_count; ++index) {
                values[index * 3u] = rgba_normal[index].x;
                values[index * 3u + 1u] = rgba_normal[index].y;
                values[index * 3u + 2u] = rgba_normal[index].z;
            }
            return true;
        }
        case 3u:
            if (destination_size != pixel_count * sizeof(uint)) {
                *error = "Instance-ID render-product readback size does not match the frame.";
                return false;
            }
            *stream_ << frame->instance_id.copy_to(
                                luisa::span{static_cast<uint*>(destination), pixel_count})
                     << synchronize();
            return true;
        case 4u:
            if (destination_size != pixel_count * sizeof(uint)) {
                *error = "Semantic-ID render-product readback size does not match the frame.";
                return false;
            }
            *stream_ << frame->semantic_id.copy_to(
                                luisa::span{static_cast<uint*>(destination), pixel_count})
                     << synchronize();
            return true;
        default:
            break;
    }
    *error = "Unknown Luisa CUDA render-product output.";
    return false;
}

} // namespace gobot::luisa_renderer
