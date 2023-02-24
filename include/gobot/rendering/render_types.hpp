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

enum class RendererType
{
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

#define BGFX_RESET_FLIP_AFTER_RENDER              UINT32_C(0x00004000)
#define BGFX_RESET_SRGB_BACKBUFFER                UINT32_C(0x00008000) //!< Enable sRGB backbuffer.
#define BGFX_RESET_HDR10                          UINT32_C(0x00010000) //!< Enable HDR10 rendering.
#define BGFX_RESET_HIDPI                          UINT32_C(0x00020000) //!< Enable HiDPI rendering.
#define BGFX_RESET_DEPTH_CLAMP                    UINT32_C(0x00040000) //!< Enable depth clamp.
#define BGFX_RESET_SUSPEND                        UINT32_C(0x00080000) //!< Suspend rendering.
#define BGFX_RESET_TRANSPARENT_BACKBUFFER         UINT32_C(0x00100000) //!< Transparent backbuffer. Availability depends on: `BGFX_CAPS_TRANSPARENT_BACKBUFFER`.

enum class RenderResetFlags : std::uint32_t {
    None                  = 0x00000000,   //!< No reset flags.
    FullScreen            = 0x00000001,   //!< Not supported yet.
    MSAA_X2               = 0x00000010,   //!< Enable 2x MSAA
    MSAA_X4               = 0x00000020,   //!< Enable 4x MSAA.
    MSAA_X8               = 0x00000030,   //!< Enable 8x MSAA.
    MSAA_X16              = 0x00000040,   //!< Enable 16x MSAA.
    Vsync                 = 0x00000080,   //!< Enable V-Sync.
    MaxAnisotropy         = 0x00000100,   //!< Turn on/off max anisotropy.
    Capture               = 0x00000200,   //!< Begin screen capture.
    FlushAfterRender      = 0x00002000,   //!< Flush rendering after submitting to GPU.
    FlapAfterRender       = 0x00004000,
    SRGB_BackBuffer       = 0x00008000,   //!< Enable sRGB backbuffer.
    HDR10                 = 0x00010000,   //!< Enable HDR10 rendering.
    HIDPI                 = 0x00020000,   //!< Enable HiDPI rendering.
    DepthClamp            = 0x00040000,   //!< Enable depth clamp.
    Suspend               = 0x00080000,   //!< Suspend rendering.
    TransparentBackBuffer = 0x00100000,   //!< Transparent backbuffer. Availability depends on: `BGFX_CAPS_TRANSPARENT_BACKBUFFER`.

    MSAAMask         = MSAA_X2 | MSAA_X4 | MSAA_X8 | MSAA_X16
};



enum class RenderDebugFlags : std::uint32_t {
    None                       = 0x00000000, //!< No debug.
    Wireframe                  = 0x00000001, //!< Enable wireframe for all primitives.
    // Enable infinitely fast hardware test. No draw calls will be submitted to driver. It's useful when profiling to quickly assess bottleneck between CPU and GPU.
    InfinitelyFastHardwareTest = 0x00000002,
    StatisticsDisplay          = 0x00000004, //!< Enable statistics display.
    DebugTextDisplay           = 0x00000008, //!< Enable debug text display.
    //!< Enable profiler. This causes per-view statistics to be collected, available through `bgfx::Stats::ViewStats`.
    //!< This is unrelated to the profiler functions in `bgfx::CallbackI`.
    Profiler                   = 0x00000010
};

enum class TextureFormat {
    BC1,          //!< DXT1 R5G6B5A1
    BC2,          //!< DXT3 R5G6B5A4
    BC3,          //!< DXT5 R5G6B5A8
    BC4,          //!< LATC1/ATI1 R8
    BC5,          //!< LATC2/ATI2 RG8
    BC6H,         //!< BC6H RGB16F
    BC7,          //!< BC7 RGB 4-7 bits per color channel, 0-8 bits alpha
    ETC1,         //!< ETC1 RGB8
    ETC2,         //!< ETC2 RGB8
    ETC2A,        //!< ETC2 RGBA8
    ETC2A1,       //!< ETC2 RGB8A1
    PTC12,        //!< PVRTC1 RGB 2BPP
    PTC14,        //!< PVRTC1 RGB 4BPP
    PTC12A,       //!< PVRTC1 RGBA 2BPP
    PTC14A,       //!< PVRTC1 RGBA 4BPP
    PTC22,        //!< PVRTC2 RGBA 2BPP
    PTC24,        //!< PVRTC2 RGBA 4BPP
    ATC,          //!< ATC RGB 4BPP
    ATCE,         //!< ATCE RGBA 8 BPP explicit alpha
    ATCI,         //!< ATCI RGBA 8 BPP interpolated alpha
    ASTC4x4,      //!< ASTC 4x4 8.0 BPP
    ASTC5x4,	  //!< ASTC 5x4 6.40 BPP
    ASTC5x5,      //!< ASTC 5x5 5.12 BPP
    ASTC6x5,	  //!< ASTC 6x5 4.27 BPP
    ASTC6x6,      //!< ASTC 6x6 3.56 BPP
    ASTC8x5,      //!< ASTC 8x5 3.20 BPP
    ASTC8x6,      //!< ASTC 8x6 2.67 BPP
    ASTC8x8,	  //!< ASTC 8x8 2.00 BPP
    ASTC10x5,     //!< ASTC 10x5 2.56 BPP
    ASTC10x6,	  //!< ASTC 10x6 2.13 BPP
    ASTC10x8,	  //!< ASTC 10x8 1.60 BPP
    ASTC10x10,	  //!< ASTC 10x10 1.28 BPP
    ASTC12x10,	  //!< ASTC 12x10 1.07 BPP
    ASTC12x12,	  //!< ASTC 12x12 0.89 BPP

    Unknown,      // Compressed formats above.

    R1,
    A8,
    R8,
    R8I,
    R8U,
    R8S,
    R16,
    R16I,
    R16U,
    R16F,
    R16S,
    R32I,
    R32U,
    R32F,
    RG8,
    RG8I,
    RG8U,
    RG8S,
    RG16,
    RG16I,
    RG16U,
    RG16F,
    RG16S,
    RG32I,
    RG32U,
    RG32F,
    RGB8,
    RGB8I,
    RGB8U,
    RGB8S,
    RGB9E5F,
    BGRA8,
    RGBA8,
    RGBA8I,
    RGBA8U,
    RGBA8S,
    RGBA16,
    RGBA16I,
    RGBA16U,
    RGBA16F,
    RGBA16S,
    RGBA32I,
    RGBA32U,
    RGBA32F,
    B5G6R5,
    R5G6B5,
    BGRA4,
    RGBA4,
    BGR5A1,
    RGB5A1,
    RGB10A2,
    RG11B10F,

    UnknownDepth, // Depth formats below.

    D16,
    D24,
    D24S8,
    D32,
    D16F,
    D24F,
    D32F,
    D0S8,

    Count
};

/// Renderer statistics data.
/// @remarks All time values are high-resolution timestamps, while
///   time frequencies define timestamps-per-second for that hardware.
using RenderStats = bgfx::Stats;


}
