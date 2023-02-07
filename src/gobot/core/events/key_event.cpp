/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-7
*/


#include "gobot/core/events/key_event.hpp"
#include "gobot/log.hpp"
#include "gobot/core/registration.hpp"
#include <magic_enum.hpp>

namespace gobot {

KeyEvent::KeyEvent(KeyCode key_code)
: key_code_(key_code) {

}

//////////////////////////////////////////////////////////////////

KeyPressedEvent::KeyPressedEvent(KeyCode key_code, uint16_t repeatCount)
    : KeyEvent(key_code),
      repeat_count_(repeatCount)
{
}

String KeyPressedEvent::ToString() const
{
    return fmt::format("KeyPressedEvent: {}({} repeats)", magic_enum::enum_name(key_code_), repeat_count_).c_str();
}

//////////////////////////////////////////////////////////////////

KeyReleasedEvent::KeyReleasedEvent(KeyCode key_code)
    : KeyEvent(key_code)
{
}

String KeyReleasedEvent::ToString() const
{
    return fmt::format("KeyReleasedEvent: {}", magic_enum::enum_name(key_code_)).c_str();
}

/////////////////////////////////////////////////////////////////

KeyTypedEvent::KeyTypedEvent(KeyCode key_code)
    : KeyEvent(key_code)
{
}

String KeyTypedEvent::ToString() const
{
    return fmt::format("KeyTypedEvent: {}", magic_enum::enum_name(key_code_)).c_str();
}

}
