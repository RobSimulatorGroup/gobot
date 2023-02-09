/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-7
*/

#pragma once

#include "gobot/core/os/window.hpp"
#include "gobot/platform/opengl/GL.hpp"

#include <gobot_export.h>
#include <GLFW/glfw3.h>

namespace gobot {

class GOBOT_EXPORT GLFWWindow : public Window {
public:
    explicit GLFWWindow(const WindowDesc& properties);

    ~GLFWWindow() override;

    bool Init(const WindowDesc& properties);

    [[nodiscard]] String GetTitle() const override;

    void SetWindowTitle(const String& title) override;

    [[nodiscard]] std::uint32_t GetWidth() const override { return window_data_.width; }

    [[nodiscard]] std::uint32_t GetHeight() const override { return window_data_.height; }

    void ToggleVSync() override;

    void SetVSync(bool v_sync) override;

    void SetBorderlessWindow(bool borderless);

    [[nodiscard]] float GetDPIScale() const override;

    [[nodiscard]] FORCE_INLINE WindowHandle GetNativeWindowHandle() const override { return native_handle_; }

    FORCE_INLINE void SetEventCallback(const EventCallbackFn& callback) override {
        window_data_.event_callback = callback;
    }

    void OnUpdate() override;

private:
    GLFWwindow* native_handle_ = nullptr;

    struct WindowData
    {
        bool v_sync = true;
        bool over_title_bar = false;
        bool exit;
        uint32_t width = 0;
        uint32_t height = 0;
        String title;
        RenderAPI render_api;
        float dpi_scale;

        EventCallbackFn event_callback;
    };


    bool init_ = false;
    WindowData window_data_;
};

}