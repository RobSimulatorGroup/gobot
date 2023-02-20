/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-8
*/

#pragma once

#include <Eigen/Dense>
#include <gobot_export.h>

namespace gobot {

struct GLPixelFormat
{
    int colorBits   = 32;
    int depthBits   = 0;
    int stencilBits = 0;
    int samples     = 0;
};

// Compare operators for GLPixelFormat structure:
bool operator == (const GLPixelFormat& lhs, const GLPixelFormat& rhs);
bool operator != (const GLPixelFormat& lhs, const GLPixelFormat& rhs);

// Base wrapper class for a GL context.
class GLContext
{

public:

    virtual ~GLContext() = default;

    // Resizes the GL context. This is called after the context surface has been resized.
    virtual void Resize(const Eigen::Vector2i& resolution) = 0;

    // Returns the number of samples for this GL context. Must be in range [1, 64].
    virtual int GetSamples() const = 0;

public:

    // Returns the color format for this GL context.
    inline Format GetColorFormat() const
    {
        return colorFormat_;
    }

    // Returns the depth-stencil format for this GL context.
    inline Format GetDepthStencilFormat() const
    {
        return depthStencilFormat_;
    }

    // Returns the state manager that is associated with this context.
    inline GLStateManager& GetStateManager()
    {
        return stateMngr_;
    }

    // Returns the global index of this GL context. This is assigned when the context is created. The first index starts with 1. The invalid index is 0.
    inline unsigned GetGlobalIndex() const
    {
        return globalIndex_;
    }

public:

    // Creates a platform specific GLContext instance.
    static std::unique_ptr<GLContext> Create(
            const GLPixelFormat&                pixelFormat,
            const RendererConfigurationOpenGL&  profile,
            Surface&                            surface,
            GLContext*                          sharedContext
    );

    // Sets the current GL context. This only stores a reference to this context (GetCurrent) and its global index (GetGlobalIndex).
    static void SetCurrent(GLContext* context);

    // Returns a pointer to the current GL context.
    static GLContext* GetCurrent();

    // Returns the global index of the current GL context ().
    static unsigned GetCurrentGlobalIndex();

    // Sets the swap interval for the current GL context.
    static bool SetCurrentSwapInterval(int interval);

protected:

    // Sets the swap interval of the platform dependent GL context.
    virtual bool SetSwapInterval(int interval) = 0;

protected:

    // Initializes the GL context with an assigned global index (GetGlobalIndex).
    GLContext();

    // Deduces the color format by the specified component bits and shifting.
    void DeduceColorFormat(int rBits, int rShift, int gBits, int gShift, int bBits, int bShift, int aBits, int aShift);

    // Deduces the depth-stencil format by the specified bit sizes.
    void DeduceDepthStencilFormat(int depthBits, int stencilBits);

    // Sets the color format to RGBA8UNorm.
    void SetDefaultColorFormat();

    // Sets the depth-stencil format to D24UNormS8UInt;
    void SetDefaultDepthStencilFormat();

private:

    GLStateManager  stateMngr_;
    Format          colorFormat_        = Format::Undefined;
    Format          depthStencilFormat_ = Format::Undefined;
    unsigned        globalIndex_        = 0;

};

}
