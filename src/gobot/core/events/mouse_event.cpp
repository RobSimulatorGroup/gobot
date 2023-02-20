/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-7
*/

#include "gobot/core/events/mouse_event.hpp"
#include "gobot/log.hpp"
#include "gobot/core/registration.hpp"
#include <magic_enum.hpp>

namespace gobot {

MouseMovedEvent::MouseMovedEvent(std::int32_t x,
                                 std::int32_t y,
                                 std::int32_t rel_x,
                                 std::int32_t rel_y,
                                 MouseButtonMask state)
    : x_(x),
      y_(y),
      rel_x_(rel_x),
      rel_y_(rel_y),
      state_(state)
{
}

String MouseMovedEvent::ToString() const
{
    return fmt::format("MouseMovedEvent: x: {0}, y: {1}, rel_x: {2}, rel_y: {3}, state: {4:#b}",
                       x_, y_, rel_x_, rel_y_, static_cast<std::uint32_t>(state_)).c_str();
}

///////////////////////////////////////////////////////////////////////


MouseScrolledEvent::MouseScrolledEvent(float x_offset, float y_offset)
    : x_offset_(x_offset),
      y_offset_(y_offset)
{
}

String MouseScrolledEvent::ToString() const
{
    return fmt::format("MouseScrolledEvent: {}, {}", x_offset_, y_offset_).c_str();
}

/////////////////////////////////////////////////////////////////////////

MouseButtonEvent::MouseButtonEvent(MouseButton button,
                                   std::int32_t x_coordinate,
                                   std::int32_t y_coordinate,
                                   MouseButtonClickMode click_mode)
    : button_(button),
      x_coordinate_(x_coordinate),
      y_coordinate_(y_coordinate),
      click_mode_(click_mode)
{
}

/////////////////////////////////////////////////////////////////////////


MouseButtonPressedEvent::MouseButtonPressedEvent(MouseButton button,
                                                 std::int32_t x_coordinate,
                                                 std::int32_t y_coordinate,
                                                 MouseButtonClickMode click_mode)
    : MouseButtonEvent(button, x_coordinate, y_coordinate, click_mode)
{
}

String MouseButtonPressedEvent::ToString() const
{
    return fmt::format("MouseButtonPressedEvent: {}", magic_enum::enum_name(button_)).c_str();
}

//////////////////////////////////////////////////////////////////////

MouseButtonReleasedEvent::MouseButtonReleasedEvent(MouseButton button,
                                                   std::int32_t x_coordinate,
                                                   std::int32_t y_coordinate,
                                                   MouseButtonClickMode click_mode)
    : MouseButtonEvent(button, x_coordinate, y_coordinate, click_mode)
{
}

String MouseButtonReleasedEvent::ToString() const
{
    return fmt::format("MouseButtonReleasedEvent: {}", magic_enum::enum_name(button_)).c_str();
}

///////////////////////////////////////////////////////////////////////


}
