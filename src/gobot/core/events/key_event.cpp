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

KeyEvent::KeyEvent(KeyCode key_code, std::uint16_t key_mod)
    : key_code_(key_code),
      key_mod_(key_mod)
{
}

//////////////////////////////////////////////////////////////////

KeyPressedEvent::KeyPressedEvent(KeyCode key_code,
                                 std::uint16_t key_mod,
                                 std::uint16_t repeat_count)
    : KeyEvent(key_code, key_mod),
      repeat_count_(repeat_count)
{
}

String KeyPressedEvent::ToString() const
{
    return fmt::format("KeyPressedEvent: {}({} repeats)", magic_enum::enum_name(key_code_), repeat_count_).c_str();
}

//////////////////////////////////////////////////////////////////

KeyReleasedEvent::KeyReleasedEvent(KeyCode key_code, std::uint16_t key_mod)
    : KeyEvent(key_code, key_mod)
{
}

String KeyReleasedEvent::ToString() const
{
    return fmt::format("KeyReleasedEvent: {}", magic_enum::enum_name(key_code_)).c_str();
}


}
