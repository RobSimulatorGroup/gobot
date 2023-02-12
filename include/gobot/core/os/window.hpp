/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-12
*/

#pragma once

#include "gobot/core/events/event.hpp"
#include "gobot/graphics/RHI/graphics_context.hpp"
#include <Eigen/Dense>

namespace gobot {

class WindowInterface {
public:
    using WindowHandle = void*;
    using EventCallbackFn = std::function<void(Event&)>;

    WindowInterface() = default;

    virtual ~WindowInterface() = default;

    [[nodiscard]] virtual std::uint32_t GetWidth() const = 0;

    [[nodiscard]] virtual std::uint32_t GetHeight() const = 0;

    [[nodiscard]]  virtual Eigen::Vector2i GetWindowSize() const = 0;

    [[nodiscard]] virtual String GetTitle() const = 0;

    virtual void SetWindowTitle(const String& title) = 0;

    [[nodiscard]] virtual bool SetWindowFullscreen() = 0;

    virtual void SetWindowBordered(bool bordered) = 0;

    [[nodiscard]] virtual bool IsWindowBordered() = 0;

    [[nodiscard]] virtual WindowHandle GetNativeWindowHandle() const = 0;

    [[nodiscard]] virtual bool IsMaximized() = 0;

    [[nodiscard]] virtual bool IsMinimized() = 0;

    virtual void Minimize() = 0;

    virtual void Maximize() = 0;

    virtual void Restore() = 0;

    [[nodiscard]] virtual bool IsFullscreen() = 0;

    virtual void RaiseWindow() = 0;

    virtual void SetIcon(const std::string& file_path, const std::string& small_icon_file_path = "") = 0;

    virtual void OnUpdate() = 0;

    virtual void UpdateCursorImGui() = 0;

//    virtual void SetEventCallback(const EventCallbackFn& callback) = 0;

protected:
//
//    SharedPtr<Lumos::Graphics::SwapChain> m_SwapChain;
//    SharedPtr<Lumos::Graphics::GraphicsContext> m_GraphicsContext;
};


}
