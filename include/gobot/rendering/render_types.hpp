/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-23
*/

#pragma once

#include <bgfx/bgfx.h>
#include <cstdint>

// This file basically the same from bgfx

namespace gobot {

enum class RendererType
{
    /// Renderer types:
    Noop,         //!< No rendering.
    Agc,          //!< AGC
    Direct3D9,    //!< Direct3D 9.0
    Direct3D11,   //!< Direct3D 11.0
    Direct3D12,   //!< Direct3D 12.0
    Gnm,          //!< GNM
    Metal,        //!< Metal
    Nvn,          //!< NVN
    OpenGLES,     //!< OpenGL ES 2.0+
    OpenGL,       //!< OpenGL 2.1+
    Vulkan,       //!< Vulkan
    WebGPU,       //!< WebGPU

    Count
};

/// Vendor PCI ID. If set to `None`, discrete and integrated GPUs will be prioritised.
enum class VendorID : std::uint16_t {
    None,                // Auto-select adapter.
    SoftWareRasterizer,  // Software rasterizer.
    AMD,                 // AMD adapter.
    Apple,               // Apple adapter.
    Intel,               // Intel adapter.
    Nvidia,              // NVIDIA adapter.
    Microsoft            // Microsoft adapter.

};


using PlatformData = bgfx::PlatformData;

using RenderInitProps = bgfx::Init;

using ViewId = bgfx::ViewId;

#define BGFX_CLEAR_NONE                           UINT16_C(0x0000) //!< No clear flags.
#define BGFX_CLEAR_COLOR                          UINT16_C(0x0001) //!< Clear color.
#define BGFX_CLEAR_DEPTH                          UINT16_C(0x0002) //!< Clear depth.
#define BGFX_CLEAR_STENCIL                        UINT16_C(0x0004) //!< Clear stencil.
#define BGFX_CLEAR_DISCARD_COLOR_0                UINT16_C(0x0008) //!< Discard frame buffer attachment 0.
#define BGFX_CLEAR_DISCARD_COLOR_1                UINT16_C(0x0010) //!< Discard frame buffer attachment 1.
#define BGFX_CLEAR_DISCARD_COLOR_2                UINT16_C(0x0020) //!< Discard frame buffer attachment 2.
#define BGFX_CLEAR_DISCARD_COLOR_3                UINT16_C(0x0040) //!< Discard frame buffer attachment 3.
#define BGFX_CLEAR_DISCARD_COLOR_4                UINT16_C(0x0080) //!< Discard frame buffer attachment 4.
#define BGFX_CLEAR_DISCARD_COLOR_5                UINT16_C(0x0100) //!< Discard frame buffer attachment 5.
#define BGFX_CLEAR_DISCARD_COLOR_6                UINT16_C(0x0200) //!< Discard frame buffer attachment 6.
#define BGFX_CLEAR_DISCARD_COLOR_7                UINT16_C(0x0400) //!< Discard frame buffer attachment 7.
#define BGFX_CLEAR_DISCARD_DEPTH                  UINT16_C(0x0800) //!< Discard frame buffer depth attachment.
#define BGFX_CLEAR_DISCARD_STENCIL                UINT16_C(0x1000) //!< Discard frame buffer stencil attachment.


enum class ClearFlags : std::uint16_t {
    None                         = 0,
    Color                        = 1 << 0,  // Clear color.
    Depth                        = 1 << 1,  // Clear depth.
    Stencil                      = 1 << 2,  // Clear stencil.
    FrameBufferColorAttachment0  = 1 << 3,  // Clear buffer attachment 0.
    FrameBufferColorAttachment1  = 1 << 4,  // Clear buffer attachment 1.
    FrameBufferColorAttachment2  = 1 << 5,  // Clear buffer attachment 2.
    FrameBufferColorAttachment3  = 1 << 6,  // Clear buffer attachment 3.
    FrameBufferColorAttachment4  = 1 << 7,  // Clear buffer attachment 4.
    FrameBufferColorAttachment5  = 1 << 8,  // Clear buffer attachment 5.
    FrameBufferColorAttachment6  = 1 << 9,  // Clear buffer attachment 6.
    FrameBufferColorAttachment7  = 1 << 10,  // Clear buffer attachment 7.
    FrameBufferDepthAttachment   = 1 << 11, // Clear buffer depth attachment.
    FrameBufferStencilAttachment = 1 << 12, // Clear buffer stencil attachment.

    FrameBufferColorAttachmentMask = FrameBufferColorAttachment0 |
                                     FrameBufferColorAttachment1 |
                                     FrameBufferColorAttachment2 |
                                     FrameBufferColorAttachment3 |
                                     FrameBufferColorAttachment4 |
                                     FrameBufferColorAttachment5 |
                                     FrameBufferColorAttachment6 | FrameBufferColorAttachment7,

    ClearAll = FrameBufferColorAttachmentMask | Color | Depth | Stencil
};


}
