/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-8
*/

#pragma once

#include <utility>
#include <Eigen/Dense>

#include "gobot/core/events/event.hpp"
#include "gobot/graphics/RHI/graphics_context.hpp"


namespace gobot {

struct WindowDesc
{
    WindowDesc(uint32_t width = 1280,
               uint32_t height = 720,
               RenderAPI render_api = RenderAPI::OpenGL,
               String title = "Gobot",
               bool fullscreen = false,
               bool vsync = true,
               bool borderless = false)
        : width(width),
          height(height),
          title(std::move(title)),
          full_screen(fullscreen),
          vsync(vsync),
          borderless(borderless),
          render_api(render_api)
    {
    }

    std::uint32_t width;
    std::uint32_t height;
    bool full_screen;
    bool vsync;
    bool borderless;
    bool show_console = true;
    String title;
    RenderAPI render_api;
};

class GOBOT_EXPORT Window
{
public:
    using WindowHandle = void*;
    using EventCallbackFn = std::function<void(Event&)>;

    Window() = default;

    virtual ~Window() = default;

    [[nodiscard]] bool HasInitialised() const { return initialised_; }

    static Window* Create(const WindowDesc& windowDesc);

    [[nodiscard]] virtual std::uint32_t GetWidth() const = 0;

    [[nodiscard]] virtual std::uint32_t GetHeight() const = 0;

    [[nodiscard]] virtual String GetTitle() const = 0;

    virtual void SetWindowTitle(const String& title) = 0;

    [[nodiscard]] virtual float GetDPIScale() const { return 1.0f; }

    [[nodiscard]] FORCE_INLINE virtual bool GetVSync() const { return v_sync_; };

    virtual void ToggleVSync() = 0;

    virtual void SetVSync(bool v_sync) = 0;

    [[nodiscard]] virtual float GetScreenRatio() const = 0;

    [[nodiscard]] virtual WindowHandle GetNativeWindowHandle() const = 0;

    [[nodiscard]] virtual bool IsMaximized() = 0;

    virtual void Minimize() = 0;

    virtual void Maximize() = 0;

    virtual void Restore() = 0;

    void SetWindowFocus(bool focus) { window_focus_ = focus; }

    [[nodiscard]] bool GetWindowFocus() const { return window_focus_; }

    virtual void SetIcon(const std::string& file_path, const std::string& small_icon_file_path = "") = 0;

    virtual void OnUpdate() = 0;

    virtual void UpdateCursorImGui() = 0;

    virtual void HideMouse(bool hide) {};

    virtual void SetEventCallback(const EventCallbackFn& callback) = 0;

protected:
    static Window* (*CreateFunc)(const WindowDesc&);

    bool initialised_ = false;
    bool v_sync_       = false;
    bool window_focus_ = true;
//
//    SharedPtr<Lumos::Graphics::SwapChain> m_SwapChain;
//    SharedPtr<Lumos::Graphics::GraphicsContext> m_GraphicsContext;
};

}
