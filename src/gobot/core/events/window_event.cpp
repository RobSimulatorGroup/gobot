/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-2-7
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/core/events/window_event.hpp"

#include <utility>
#include "gobot/log.hpp"

namespace gobot {


WindowResizeEvent::WindowResizeEvent(std::uint32_t width, std::uint32_t height)
    : width_(width),
      height_(height)
{
}

std::string WindowResizeEvent::ToString() const
{
    return fmt::format("WindowResizeEvent: {}, {}", width_, height_);
}

////////////////////////////////////////////////////

WindowMovedEvent::WindowMovedEvent(std::uint32_t x, std::uint32_t y)
    : x_(x),
      y_(y)
{
}

std::string WindowMovedEvent::ToString() const {
    return fmt::format("WindowMovedEvent: ({}, {})", x_, y_);
}

////////////////////////////////////////////

WindowDropFileEvent::WindowDropFileEvent(std::string file_path)
    : file_path_(std::move(file_path))
{
}

std::string WindowDropFileEvent::ToString() const {
    return fmt::format("WindowFileEvent: {}", file_path_);
}



}