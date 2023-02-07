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

MouseMovedEvent::MouseMovedEvent(float x, float y)
    : mouse_x_(x),
      mouse_y_(y)
{
}

String MouseMovedEvent::ToString() const
{
    return fmt::format("MouseMovedEvent: {}, {}", mouse_x_, mouse_y_).c_str();
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

MouseButtonEvent::MouseButtonEvent(MouseKeyCode button)
    : button_(button)
{
}

/////////////////////////////////////////////////////////////////////////


MouseButtonPressedEvent::MouseButtonPressedEvent(MouseKeyCode button)
    : MouseButtonEvent(button)
{
}

String MouseButtonPressedEvent::ToString() const
{
    return fmt::format("MouseButtonPressedEvent: {}", magic_enum::enum_name(button_)).c_str();
}

//////////////////////////////////////////////////////////////////////

MouseButtonReleasedEvent::MouseButtonReleasedEvent(MouseKeyCode button)
    : MouseButtonEvent(button)
{
}

String MouseButtonReleasedEvent::ToString() const
{
    return fmt::format("MouseButtonReleasedEvent: {}", magic_enum::enum_name(button_)).c_str();
}


}
