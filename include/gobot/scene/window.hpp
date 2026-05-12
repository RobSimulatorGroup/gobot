/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-1-14
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/scene/node.hpp"
#include "gobot/core/events/window_event.hpp"
#include "gobot/core/events/event.hpp"
#include "gobot/core/os/window.hpp"

#include <functional>

namespace gobot {


class GOBOT_EXPORT Window : public Node {
    GOBCLASS(Window, Node);
public:
    using WindowFactory = std::function<std::unique_ptr<WindowInterface>()>;

    Window(bool p_init_sdl_window = true);

    ~Window() override;

    static void SetWindowFactory(WindowFactory factory);

    void SetVisible(bool visible);

    bool IsVisible() const;

    void OnEvent(Event& e);

    void PullEvent();

    WindowInterface* GetWindow() { return window_.get(); }

    void SwapBuffers();

private:

    std::unique_ptr<WindowInterface> window_{nullptr};
};

}
