/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-2-12
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/io/image.hpp"
#include "gobot/core/events/event.hpp"
#include <Eigen/Dense>

namespace gobot {

class WindowInterface {
public:
    using NativeWindowHandle = void*;

    using EventCallbackFn = std::function<void(Event&)>;

    WindowInterface() = default;

    virtual ~WindowInterface() = default;

    [[nodiscard]] virtual std::uint32_t GetWidth() const = 0;

    [[nodiscard]] virtual std::uint32_t GetHeight() const = 0;

    [[nodiscard]]  virtual Eigen::Vector2i GetWindowSize() const = 0;

    [[nodiscard]] virtual std::string GetTitle() const = 0;

    virtual void SetTitle(const std::string& title) = 0;

    [[nodiscard]] virtual bool SetWindowFullscreen() = 0;

    virtual void SetWindowBordered(bool bordered) = 0;

    [[nodiscard]] virtual bool IsWindowBordered() = 0;

    virtual void ShowWindow() = 0;

    virtual void HideWindow() = 0;

    virtual bool IsWindowHide() = 0;

    [[nodiscard]] virtual NativeWindowHandle GetNativeWindowHandle() const = 0;

    [[nodiscard]] virtual void* GetNativeDisplayHandle() const = 0;

    [[nodiscard]] virtual bool IsMaximized() = 0;

    [[nodiscard]] virtual bool IsMinimized() = 0;

    virtual void Minimize() = 0;

    virtual void Maximize() = 0;

    virtual void Restore() = 0;

    [[nodiscard]] virtual bool IsFullscreen() = 0;

    virtual void RaiseWindow() = 0;

    virtual void SetIcon(const Ref<Image>& image) = 0;

    virtual void ProcessEvents() = 0;

    virtual void SwapBuffers() = 0;

protected:

    bool render_need_reset_{false};

};


}
