/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/rendering/headless_render_context.hpp"

#include "gobot/drivers/opengl/rasterizer_gl.hpp"
#include "gobot/log.hpp"
#include "gobot/rendering/render_server.hpp"

#include <memory>

#if defined(GOBOT_HAS_EGL)
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <glad/glad.h>
#endif

namespace gobot {

struct HeadlessRenderContext::PlatformContext {
#if defined(GOBOT_HAS_EGL)
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface surface = EGL_NO_SURFACE;
#endif
};

namespace {

#if defined(GOBOT_HAS_EGL)
const char* EglErrorName(EGLint error) {
    switch (error) {
        case EGL_SUCCESS:
            return "EGL_SUCCESS";
        case EGL_NOT_INITIALIZED:
            return "EGL_NOT_INITIALIZED";
        case EGL_BAD_ACCESS:
            return "EGL_BAD_ACCESS";
        case EGL_BAD_ALLOC:
            return "EGL_BAD_ALLOC";
        case EGL_BAD_ATTRIBUTE:
            return "EGL_BAD_ATTRIBUTE";
        case EGL_BAD_CONTEXT:
            return "EGL_BAD_CONTEXT";
        case EGL_BAD_CONFIG:
            return "EGL_BAD_CONFIG";
        case EGL_BAD_CURRENT_SURFACE:
            return "EGL_BAD_CURRENT_SURFACE";
        case EGL_BAD_DISPLAY:
            return "EGL_BAD_DISPLAY";
        case EGL_BAD_SURFACE:
            return "EGL_BAD_SURFACE";
        case EGL_BAD_MATCH:
            return "EGL_BAD_MATCH";
        case EGL_BAD_PARAMETER:
            return "EGL_BAD_PARAMETER";
        case EGL_BAD_NATIVE_PIXMAP:
            return "EGL_BAD_NATIVE_PIXMAP";
        case EGL_BAD_NATIVE_WINDOW:
            return "EGL_BAD_NATIVE_WINDOW";
        case EGL_CONTEXT_LOST:
            return "EGL_CONTEXT_LOST";
        default:
            return "EGL_UNKNOWN_ERROR";
    }
}

std::string CurrentEglError(const char* action) {
    const EGLint error = eglGetError();
    return std::string(action) + " failed: " + EglErrorName(error);
}
#endif

}

HeadlessRenderContext::HeadlessRenderContext() = default;

HeadlessRenderContext::~HeadlessRenderContext() {
    render_server_.reset();

#if defined(GOBOT_HAS_EGL)
    if (platform_) {
        if (platform_->display != EGL_NO_DISPLAY) {
            eglMakeCurrent(platform_->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            if (platform_->context != EGL_NO_CONTEXT) {
                eglDestroyContext(platform_->display, platform_->context);
            }
            if (platform_->surface != EGL_NO_SURFACE) {
                eglDestroySurface(platform_->display, platform_->surface);
            }
            eglTerminate(platform_->display);
        }
    }
#endif
}

bool HeadlessRenderContext::Initialize() {
    if (IsReady()) {
        return true;
    }

    if (RenderServer::HasInstance()) {
        last_error_.clear();
        return true;
    }

#if !defined(GOBOT_HAS_EGL)
    last_error_ = "Gobot was built without EGL headless rendering support.";
    return false;
#else
    platform_ = std::make_unique<PlatformContext>();

    auto get_platform_display =
            reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));
    if (get_platform_display != nullptr) {
        platform_->display = get_platform_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
    }
    if (platform_->display == EGL_NO_DISPLAY) {
        platform_->display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    }
    if (platform_->display == EGL_NO_DISPLAY) {
        last_error_ = CurrentEglError("eglGetDisplay");
        return false;
    }

    EGLint major = 0;
    EGLint minor = 0;
    if (eglInitialize(platform_->display, &major, &minor) != EGL_TRUE) {
        last_error_ = CurrentEglError("eglInitialize");
        return false;
    }

    if (eglBindAPI(EGL_OPENGL_API) != EGL_TRUE) {
        last_error_ = CurrentEglError("eglBindAPI");
        return false;
    }

    const EGLint config_attributes[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_RED_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_BLUE_SIZE, 8,
            EGL_ALPHA_SIZE, 8,
            EGL_DEPTH_SIZE, 24,
            EGL_NONE,
    };

    EGLConfig config = nullptr;
    EGLint config_count = 0;
    if (eglChooseConfig(platform_->display, config_attributes, &config, 1, &config_count) != EGL_TRUE ||
        config_count <= 0) {
        last_error_ = CurrentEglError("eglChooseConfig");
        return false;
    }

    const EGLint pbuffer_attributes[] = {
            EGL_WIDTH, 1,
            EGL_HEIGHT, 1,
            EGL_NONE,
    };
    platform_->surface = eglCreatePbufferSurface(platform_->display, config, pbuffer_attributes);
    if (platform_->surface == EGL_NO_SURFACE) {
        last_error_ = CurrentEglError("eglCreatePbufferSurface");
        return false;
    }

    constexpr EGLint versions[][2] = {
            {4, 6},
            {4, 5},
            {4, 3},
            {3, 3},
    };
    for (const auto& version : versions) {
        const EGLint context_attributes[] = {
                EGL_CONTEXT_MAJOR_VERSION, version[0],
                EGL_CONTEXT_MINOR_VERSION, version[1],
                EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                EGL_NONE,
        };
        platform_->context = eglCreateContext(platform_->display, config, EGL_NO_CONTEXT, context_attributes);
        if (platform_->context != EGL_NO_CONTEXT) {
            break;
        }
    }
    if (platform_->context == EGL_NO_CONTEXT) {
        last_error_ = CurrentEglError("eglCreateContext");
        return false;
    }

    if (eglMakeCurrent(platform_->display, platform_->surface, platform_->surface, platform_->context) != EGL_TRUE) {
        last_error_ = CurrentEglError("eglMakeCurrent");
        return false;
    }

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(eglGetProcAddress))) {
        last_error_ = "Failed to initialize GLAD from EGL.";
        return false;
    }

    opengl::GLRasterizer::MakeCurrent();
    render_server_ = std::make_unique<RenderServer>();

    const GLubyte* vendor = glGetString(GL_VENDOR);
    const GLubyte* renderer = glGetString(GL_RENDERER);
    const GLubyte* version = glGetString(GL_VERSION);
    LOG_INFO("EGL headless OpenGL loaded: vendor='{}', renderer='{}', version='{}'",
             vendor != nullptr ? reinterpret_cast<const char*>(vendor) : "",
             renderer != nullptr ? reinterpret_cast<const char*>(renderer) : "",
             version != nullptr ? reinterpret_cast<const char*>(version) : "");

    last_error_.clear();
    return true;
#endif
}

bool HeadlessRenderContext::IsReady() const {
    return RenderServer::HasInstance();
}

const std::string& HeadlessRenderContext::GetLastError() const {
    return last_error_;
}

}
