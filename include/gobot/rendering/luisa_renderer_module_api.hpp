/*
 * Private ABI between Gobot and the optional LuisaCompute renderer module.
 * Both sides must be built from the same Gobot source revision.
 */

#pragma once

#include "gobot/rendering/renderer_scene_render.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace gobot {

inline constexpr std::uint32_t GOBOT_LUISA_RENDERER_ABI_VERSION = 3;

struct LuisaRendererTarget {
    std::uint32_t gl_color_texture = 0;
    std::uint32_t gl_depth_texture = 0;
    int width = 0;
    int height = 0;
};

struct LuisaRenderProductRequest {
    int width = 0;
    int height = 0;
    std::uint32_t output_mask = 0;
    std::uint32_t mode = 0;
};

struct LuisaRenderProductBuffer {
    void* device_pointer = nullptr;
    std::size_t allocation_size = 0;
    std::size_t pixel_stride_bytes = 0;
};

struct LuisaRenderProductFrame {
    void* frame = nullptr;
    std::array<LuisaRenderProductBuffer, 5> buffers{};
    int device_id = 0;
};

enum class LuisaRendererResult : std::uint32_t {
    Success,
    Unavailable,
    RecoverableError,
    FatalError
};

struct LuisaRendererModuleApi {
    std::uint32_t abi_version = 0;
    void* (*create)(const char* module_directory, char* error, std::size_t error_size) = nullptr;
    void (*destroy)(void* renderer) = nullptr;
    SceneRendererCapabilities (*capabilities)(void* renderer) = nullptr;
    LuisaRendererResult (*render)(void* renderer,
                                  const LuisaRendererTarget* target,
                                  const RenderSceneSnapshot* scene,
                                  const RenderViewSnapshot* view,
                                  const SceneRendererSettings* settings,
                                  SceneRendererStats* stats,
                                  char* error,
                                  std::size_t error_size) = nullptr;
    void (*reset_accumulation)(void* renderer) = nullptr;
    LuisaRendererResult (*capture_render_product)(void* renderer,
                                                  const RenderSceneSnapshot* scene,
                                                  const RenderViewSnapshot* view,
                                                  const LuisaRenderProductRequest* request,
                                                  LuisaRenderProductFrame* frame,
                                                  char* error,
                                                  std::size_t error_size) = nullptr;
    void (*release_render_product)(void* renderer, void* frame) = nullptr;
    bool (*readback_render_product)(void* renderer,
                                    void* frame,
                                    std::uint32_t output,
                                    void* destination,
                                    std::size_t destination_size,
                                    char* error,
                                    std::size_t error_size) = nullptr;
};

using GetLuisaRendererModuleApi = const LuisaRendererModuleApi* (*)();

} // namespace gobot

extern "C" const gobot::LuisaRendererModuleApi* gobot_luisa_renderer_get_api();
