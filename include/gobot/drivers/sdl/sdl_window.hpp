/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-11
*/

#pragma once

#include "gobot/core/os/window.hpp"

#include <Eigen/Dense>

class SDL_Window;

namespace gobot {

class GOBOT_EXPORT SDLWindow final: public WindowInterface {
public:
    SDLWindow();

    ~SDLWindow() override;

    void SetWindowPosition(const Eigen::Vector2i& position) override;

    [[nodiscard]] Eigen::Vector2i GetWindowPosition() const override;

    [[nodiscard]] std::uint32_t GetWidth() const override;

    [[nodiscard]] std::uint32_t GetHeight() const override;

    [[nodiscard]] Eigen::Vector2i GetWindowSize() const override;

    void SetWindowSize(const Eigen::Vector2i& size) override;

    [[nodiscard]] bool SetWindowFullscreen() override;

    [[nodiscard]] bool IsFullscreen() override;

    [[nodiscard]] bool IsWindowBordered() override;

    void SetWindowBordered(bool bordered) override;

    void ShowWindow() override;

    void HideWindow() override;

    [[nodiscard]] bool IsWindowHide() const override;

    [[nodiscard]] String GetWindowTitle() const override;

    void SetWindowTitle(const String& title) override;

    [[nodiscard]] bool IsMaximized() override;

    [[nodiscard]] bool IsMinimized() override;

    // Make a window as large as possible.
    void Maximize() override;

    // Minimize a window to an iconic representation.
    void Minimize() override;

    // Restore the size and position of a minimized or maximized window.
    void Restore() override;

    // Raise a window above other windows and set the input focus.
    void RaiseWindow() override;

    void SetIcon(const Ref<Image>& image) override;

    [[nodiscard]] std::uint32_t GetWindowID() const;

    void PollEvents() override;

    [[nodiscard]] WindowHandle GetNativeWindowHandle() const override;

    void SetEventCallback(const EventCallbackFn& callback) override { event_callback_ = callback; }

    void RunEventCallback(Event& event);


private:
    SDL_Window* native_window_{nullptr};

    EventCallbackFn event_callback_{nullptr};
    std::uint32_t windows_id_; // cache of sdl windows id
};


}
