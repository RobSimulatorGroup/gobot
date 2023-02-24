/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-23
*/

#pragma once

#include "gobot/core/object.hpp"
#include "gobot/core/color.hpp"
#include "gobot/rendering/render_types.hpp"

namespace gobot {

#define GET_RENDER_SERVER()     \
    RenderServer::GetInstance()


class RenderServer : public Object {
    GOBCLASS(RenderServer, Object)
public:
    RenderServer();

    ~RenderServer() override;

    static RenderServer* GetInstance();

    // Initialize the renderer.
    void InitWindow();

    void SetViewClear(ViewId view_id,
                      ClearFlags clear_flags,
                      const Color& color = {0.f, 0.f, 0.f, 1.0},
                      float depth = 1.0f,
                      uint8_t stencil = 0);

    void SetViewRect(ViewId id, uint16_t x, uint16_t y, uint16_t width, uint16_t height);

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
    ///
    /// @attention C99's equivalent binding is `bgfx_reset`.
    ///
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

private:
    static RenderServer* s_singleton;

};

}
