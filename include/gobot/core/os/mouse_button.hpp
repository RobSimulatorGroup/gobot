/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-11
*/

#pragma once

namespace gobot {

// Same as SDL_MouseButtonEvent
enum class MouseButtonClickMode : std::uint8_t
{
    Single = 1,
    Double = 2
};

// Same as SDL_MouseButtonEvent
enum class MouseButton : std::uint8_t
{
    Left      = 1,
    Middle    = 2,
    Right     = 3,
    X1        = 4,
    X2        = 5,

    ButtonMaxNum = 8 /**< not a button, just marks the number of buttons
                                 for array bounds */
};

// Same as SDL_MouseButtonEvent
enum class MouseButtonMask : std::uint32_t
{
    Left      = 1 << (static_cast<std::uint8_t>(MouseButton::Left) - 1),
    Middle    = 1 << (static_cast<std::uint8_t>(MouseButton::Middle) - 1),
    Right     = 1 << (static_cast<std::uint8_t>(MouseButton::Right) - 1),
    X1        = 1 << (static_cast<std::uint8_t>(MouseButton::X1) - 1),
    X2        = 1 << (static_cast<std::uint8_t>(MouseButton::X2) - 1),
};

}  // end of namespace gobot

