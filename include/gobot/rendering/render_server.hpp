/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-23
*/

#pragma once

#include "gobot/core/object.hpp"
#include "gobot/core/color.hpp"
#include "gobot/core/math/matrix.hpp"
#include "gobot/rendering/render_types.hpp"
#include "gobot/rendering/render_rid.hpp"

namespace gobot {

#define GET_RS()     \
    RenderServer::GetInstance()

class GOBOT_EXPORT RenderServer : public Object {
    GOBCLASS(RenderServer, Object)
public:
    RenderServer();

    ~RenderServer() override;

    static bool HasInit();

    RendererType GetRendererType();

    static RenderServer* GetInstance();

    // Initialize the renderer.
    void InitWindow();

    void ShutDown();

    // Debug related
    void SetDebug(RenderDebugFlags debug_flags);

    void DebugTextClear();

    /// @param[in] x, y 2D position from top-left.
    /// @param[in] width, height  Image width and height.
    /// @param[in] data  Raw image data (character/attribute raw encoding).
    /// @param[in] pitch Image pitch in bytes.
    void DebugTextImage(uint16_t x, uint16_t y, uint16_t width, uint16_t height, const void* data, uint16_t pitch);

    void DebugTextPrintf(uint16_t x, uint16_t y, uint8_t attr, const char* format, ...);

    /// Advance to next frame. When using multithreaded renderer, this call
    /// just swaps internal buffers, kicks render thread, and returns. In
    /// singlethreaded renderer this call does frame rendering.
    ///
    /// @param[in] capture Capture frame with graphics debugger.
    ///
    /// @returns Current frame number. This might be used in conjunction with
    ///   double/multi buffering data outside the library and passing it to
    ///   library via `bgfx::makeRef` calls.
    uint32_t Frame(bool capture = false);

    /// Reset graphic settings and back-buffer size.
    ///
    /// @param[in] width Back-buffer width.
    /// @param[in] height Back-buffer height.
    /// @param[in] flags See: `RenderResetFlags` for more info.
    ///   - `None` - No reset flags.
    ///   - `FullScreen` - Not supported yet.
    ///   - `MSAA_X[2/4/8/16]` - Enable 2, 4, 8 or 16 x MSAA.
    ///   - `Vsync` - Enable V-Sync.
    ///   - `MaxAnisotropy` - Turn on/off max anisotropy.
    ///   - `Capture` - Begin screen capture.
    ///   - `FlushAfterRender` - Flush rendering after submitting to GPU.
    ///   - `FlapAfterRender` - This flag  specifies where flip
    ///     occurs. Default behavior is that flip occurs before rendering new
    ///     frame. This flag only has effect when `BGFX_CONFIG_MULTITHREADED=0`.
    ///   - `BGFX_RESET_SRGB_BACKBUFFER` - Enable sRGB back-buffer.
    /// @param[in] _format Texture format. See: `TextureFormat::Enum`.
    ///
    /// @attention This call doesn’t change the window size, it just resizes
    ///   the back-buffer. Your windowing code controls the window size.
    void Reset(uint32_t _width,
               uint32_t _height,
               RenderResetFlags reset_flags = RenderResetFlags::None,
               TextureFormat format = TextureFormat::Count);

    /// Submit an empty primitive for rendering. Uniforms and draw state
    /// will be applied but no geometry will be submitted.
    ///
    /// These empty draw calls will sort before ordinary draw calls.
    ///
    /// @param[in] id View id.
    void Touch(ViewId id);

    const RenderStats* GetStats();

    FORCE_INLINE const RenderDebugFlags GetDebugFlags() const { return debug_flags_; }

    FORCE_INLINE const RenderResetFlags GetResetFlags() const { return reset_flags_; }

    FORCE_INLINE void SetResetFlags(RenderResetFlags reset_flags) { reset_flags_ = reset_flags; }

    FORCE_INLINE void SetDebugFlags(RenderDebugFlags debug_flags) { debug_flags_ = debug_flags; }

    // render texture

    /// Create 2D texture.
    ///
    /// @param[in] width Width.
    /// @param[in] height Height.
    /// @param[in] has_mips Indicates that texture contains full mip-map chain.
    /// @param[in] num_layers Number of layers in texture array. Must be 1 if caps
    ///   `BGFX_CAPS_TEXTURE_2D_ARRAY` flag is not set.
    /// @param[in] format Texture format. See: `TextureFormat`.
    /// @param[in] flags Texture creation
    ///   flags. Default texture sampling mode is linear, and wrap mode is repeat.
    ///   - `Sampler_[U/V/W]_[Mirror/Clamp]` - Mirror or clamp to edge wrap mode.
    ///   - `Sampler_[Min/Mag/Mip]_[Point/Anisotropy]` - Point or anisotropic sampling.
    ///
    /// @param[in] mem Texture data. If `mem` is non-NULL, created texture will be immutable. If
    ///   `mem` is NULL content of the texture is uninitialized. When `num_layers` is more than 1,
    /// expected memory layout is texture and all mips together for each array element.
    RenderRID CreateTexture2D(uint16_t width,
                              uint16_t height,
                              bool has_mips,
                              uint16_t num_layers,
                              TextureFormat format,
                              TextureFlags flags,
                              const MemoryView* mem = nullptr);

    RenderRID CreateTexture3D(uint16_t width,
                              uint16_t height,
                              uint16_t depth,
                              bool has_mips,
                              TextureFormat format,
                              TextureFlags flags,
                              const MemoryView* mem = nullptr);

    RenderRID CreateTextureCube(uint16_t size,
                                bool has_mips,
                                uint16_t num_layers,
                                TextureFormat format,
                                TextureFlags flags,
                                const MemoryView* mem = nullptr);

    bool FreeTexture(const RenderRID& rid);

    RenderRID CreateMesh();

    bool FreeMesh(const RenderRID& rid);

private:
    static RenderServer* s_singleton;

    RenderDebugFlags debug_flags_;
    RenderResetFlags reset_flags_;
};

}
