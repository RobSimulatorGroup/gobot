/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-8
*/

#pragma once

#include <utility>

#include "gobot/core/events/event.hpp"

namespace gobot {

struct WindowDesc
{
    WindowDesc(uint32_t width = 1280,
               uint32_t height = 720,
               int renderAPI = 0,
               String title = "Gobot",
               bool fullscreen = false,
               bool vsync = true,
               bool borderless = false)
        : width_(width),
          height_(height),
          title_(std::move(title)),
          full_screen_(fullscreen),
          vsync_(vsync),
          borderless_(borderless),
          RenderAPI(renderAPI)
        {
        }

    std::uint32_t width_;
    std::uint32_t height_;
    bool full_screen_;
    bool vsync_;
    bool borderless_;
    bool show_console_ = true;
    String title_;
    int RenderAPI;
};

class Window
{
public:
    using WindowHandle = void*;
    using EventCallbackFn = std::function<void(Event&)>;

    virtual ~Window() = default;

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

//    virtual void SetIcon(const std::string& filePath, const std::string& smallIconFilePath = "") = 0;

    virtual void OnUpdate();

    virtual void SetEventCallback(const EventCallbackFn& callback);

protected:
    static Window* (*CreateFunc)(const WindowDesc&);

    Window() = default;

//    bool init_ = false;
//    glm::vec2 m_Position;
    bool v_sync_       = false;
//    bool m_HasResized  = false;
//    bool m_WindowFocus = true;
//
//    SharedPtr<Lumos::Graphics::SwapChain> m_SwapChain;
//    SharedPtr<Lumos::Graphics::GraphicsContext> m_GraphicsContext;
};

}
