///* The gobot is a robot simulation platform.
// * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
// * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
// * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
// * This file is created by Qiqi Wu, 23-1-14
//*/

#pragma once

#include "gobot/scene/node.hpp"
#include "gobot/core/events/window_event.hpp"
#include "gobot/core/events/event.hpp"
#include "gobot/render/graphics_context.hpp"
#include "gobot/core/os/window.hpp"

namespace gobot {

enum class WindowDriver {
    SDL,            // Default(already cross-platform for all platforms)
    Win32,     // not implemented
    IOSWindow       // not implemented
};


class GOBOT_EXPORT Window : public Node {
    Q_OBJECT
    GOBCLASS(Window, Node);
public:
    Window();

    ~Window() override;

    void SetVisible(bool visible);

    bool IsVisible() const;

    void OnEvent(Event& e);

    void PullEvent();

Q_SIGNALS:
    void windowCloseRequested();
    void windowResizeRequested(WindowResizeEvent& event);
    void windowMaximizedRequested();
    void windowMinimizedRequested();
    void windowMovedRequested();
    void windowTakeFocusRequested();
    void windowDropFileRequested();

private:
    void RegisterWindowCallbacks();

private:

    WindowDriver window_driver_{WindowDriver::SDL};
    std::unique_ptr<WindowInterface> window_interface_{nullptr};
};

}