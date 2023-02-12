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

class SDLWindow : public WindowInterface {
public:
    SDLWindow();

    ~SDLWindow() override;

    [[nodiscard]] std::uint32_t GetWidth() const override;

    [[nodiscard]] std::uint32_t GetHeight() const override;

    [[nodiscard]] Eigen::Vector2i GetWindowSize() const override;

    [[nodiscard]] String GetTitle() const override;

    void SetWindowTitle(const String& title) override;

    [[nodiscard]] bool IsMaximized() override;

    [[nodiscard]] bool IsMinimized() override;

    void Maximize() override;

    void Minimize() override;

    void Restore() override;

    void RaiseWindow() override;

    virtual void SetIcon(const std::string& file_path, const std::string& small_icon_file_path = "") {};

    void UpdateCursorImGui();

    void OnUpdate();

    [[nodiscard]] WindowHandle GetNativeWindowHandle() const override;

private:
//    void Init();

    RenderAPI render_api_;
    SDL_Window* native_window_{nullptr};
};


}
