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

namespace gobot {

#define GET_RENDER_SERVER()     \
    RenderServer::GetInstance()


class RenderServer : public Object {
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

    /// Set view's view matrix and projection matrix,
    /// all draw primitives in this view will use these two matrices.
    ///
    /// @param[in] id View id.
    /// @param[in] view View matrix(ColMajor).
    /// @param[in] proj Projection matrix(ColMajor).
    void SetViewTransform(ViewId id, const Matrix4f& view, const Matrix4f& proj);

    void SetViewClear(ViewId view_id,
                      RenderClearFlags clear_flags,
                      const Color& color = {0.f, 0.f, 0.f, 1.0},
                      float depth = 1.0f,
                      uint8_t stencil = 0);

    void SetViewRect(ViewId id, uint16_t x, uint16_t y, uint16_t width, uint16_t height);

    // Debug related
    void SetDebug(RenderDebugFlags debug_flags);

    /// Request screen shot of main window back buffer.
    /// @param[in] file_path Will be passed to `bgfx::CallbackI::screenShot` callback.
    void RequestScreenShot(const String& file_path);

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



    // Render related
    void SetIndexBuffer(IndexBufferHandle buffer_handle);


    void SetVertexBuffer(uint8_t stream, VertexBufferHandle handle);

    /// Set shader uniform parameter for draw primitive.
    ///
    /// @param[in] handle Uniform.
    /// @param[in] value Pointer to uniform data.
    /// @param[in] num Number of elements. Passing `UINT16_MAX` will use the num passed on uniform creation.
    void SetUniform(UniformHandle handle, const void* value, uint16_t num = 1);

    /// Set render states for draw primitive.
    /// WriteRGB | WriteAlpha | WriteDepth | DepthTestLess | CullCW | MSAA
    /// @param[in] state_flags State flags. Default state for primitive type is
    ///   triangles. See: `BGFX_STATE_DEFAULT`.
    ///   - `DepthTest*` - Depth test function.
    ///   - `Blend*` - See remark 1 about BGFX_STATE_BLEND_FUNC.
    ///   - `BGFX_STATE_BLEND_EQUATION_*` - See remark 2.
    ///   - `BGFX_STATE_CULL_*` - Backface culling mode.
    ///   - `BGFX_STATE_WRITE_*` - Enable R, G, B, A or Z write.
    ///   - `BGFX_STATE_MSAA` - Enable MSAA.
    ///   - `BGFX_STATE_PT_[TRISTRIP/LINES/POINTS]` - Primitive type.
    ///
    /// @param[in] rgba Sets blend factor used by `BGFX_STATE_BLEND_FACTOR` and
    ///   `BGFX_STATE_BLEND_INV_FACTOR` blend modes.
    ///
    /// @remarks
    ///   1. To set up more complex states use:
    ///      `BGFX_STATE_ALPHA_REF(_ref)`,
    ///      `BGFX_STATE_POINT_SIZE(_size)`,
    ///      `BGFX_STATE_BLEND_FUNC(_src, _dst)`,
    ///      `BGFX_STATE_BLEND_FUNC_SEPARATE(_srcRGB, _dstRGB, _srcA, _dstA)`
    ///      `BGFX_STATE_BLEND_EQUATION(_equation)`
    ///      `BGFX_STATE_BLEND_EQUATION_SEPARATE(_equationRGB, _equationA)`
    ///   2. `BGFX_STATE_BLEND_EQUATION_ADD` is set when no other blend
    ///      equation is specified.
    void SetState(RenderStateFlags state_flags, uint32_t rgba = 0);

    /// Submit primitive for rendering.
    ///
    /// @param[in] id View id.
    /// @param[in] program Program.
    /// @param[in] depth Depth for sorting.
    /// @param[in] flags Discard or preserve states.
    void Submit(ViewId id, ProgramHandle program, uint32_t depth = 0, RenderEncoderDiscardFlags flags = RenderEncoderDiscardFlags::All);


    ShaderHandle  CreateShader(const ShaderMemory *mem);

    ProgramHandle CreateProgram(ShaderHandle vsh, ShaderHandle fsh, bool destroy_shaders = false);

private:
    static RenderServer* s_singleton;

    RenderDebugFlags debug_flags_;
    RenderResetFlags reset_flags_;
};

}
