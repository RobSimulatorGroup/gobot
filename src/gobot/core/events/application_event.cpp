/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-7
*/

#include "gobot/core/events/application_event.hpp"

#include <utility>
#include "gobot/log.hpp"

namespace gobot {


WindowResizeEvent::WindowResizeEvent(std::uint32_t width, std::uint32_t height)
    : width_(width),
      height_(height),
      dpi_scale_(1.0)
{
}

WindowResizeEvent::WindowResizeEvent(std::uint32_t width, std::uint32_t height, float dpi_scale)
    : width_(width),
      height_(height),
      dpi_scale_(dpi_scale)
{
}

WindowFileEvent::WindowFileEvent(String file_path)
    : file_path_(std::move(file_path))
{
}

String WindowFileEvent::ToString() const {
    return fmt::format("WindowFileEvent: {}", file_path_).c_str();
}

String WindowResizeEvent::ToString() const
{
    return fmt::format("WindowResizeEvent: {}, {}", width_, height_).c_str();
}


}