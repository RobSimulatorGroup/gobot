/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-8
*/

#pragma once

#include <SDL.h>

namespace gobot {

//static KeyCode GLFWToGobotKeyboardKey(uint32_t glfwKey)
//{
//    static std::map<uint32_t, KeyCode> s_key_map = {
//            /* Printable keys */
//            { GLFW_KEY_SPACE, KeyCode::Space },
//            { GLFW_KEY_APOSTROPHE, KeyCode::Apostrophe }, /* ' */
//            { GLFW_KEY_COMMA, KeyCode::Comma }, /* , */
//            { GLFW_KEY_MINUS, KeyCode::Minus },  /* - */
//            { GLFW_KEY_PERIOD, KeyCode::Period },  /* . */
//            { GLFW_KEY_SLASH, KeyCode::Slash }, /* / */
//
//            { GLFW_KEY_0, KeyCode::D0 },
//            { GLFW_KEY_1, KeyCode::D1 },
//            { GLFW_KEY_2, KeyCode::D2 },
//            { GLFW_KEY_3, KeyCode::D3 },
//            { GLFW_KEY_4, KeyCode::D4 },
//            { GLFW_KEY_5, KeyCode::D5 },
//            { GLFW_KEY_6, KeyCode::D6 },
//            { GLFW_KEY_7, KeyCode::D7 },
//            { GLFW_KEY_8, KeyCode::D8 },
//            { GLFW_KEY_9, KeyCode::D9 },
//
//            { GLFW_KEY_SEMICOLON, KeyCode::Semicolon }, /* ; */
//            { GLFW_KEY_EQUAL, KeyCode::Equal },  /* = */
//
//            { GLFW_KEY_A, KeyCode::A },
//            { GLFW_KEY_B, KeyCode::B },
//            { GLFW_KEY_C, KeyCode::C },
//            { GLFW_KEY_D, KeyCode::D },
//            { GLFW_KEY_E, KeyCode::E },
//            { GLFW_KEY_F, KeyCode::F },
//            { GLFW_KEY_G, KeyCode::G },
//            { GLFW_KEY_H, KeyCode::H },
//            { GLFW_KEY_I, KeyCode::I },
//            { GLFW_KEY_J, KeyCode::J },
//            { GLFW_KEY_K, KeyCode::K },
//            { GLFW_KEY_L, KeyCode::L },
//            { GLFW_KEY_M, KeyCode::M },
//            { GLFW_KEY_N, KeyCode::N },
//            { GLFW_KEY_O, KeyCode::O },
//            { GLFW_KEY_P, KeyCode::P },
//            { GLFW_KEY_Q, KeyCode::Q },
//            { GLFW_KEY_R, KeyCode::R },
//            { GLFW_KEY_S, KeyCode::S },
//            { GLFW_KEY_T, KeyCode::T },
//            { GLFW_KEY_U, KeyCode::U },
//            { GLFW_KEY_V, KeyCode::V },
//            { GLFW_KEY_W, KeyCode::W },
//            { GLFW_KEY_X, KeyCode::X },
//            { GLFW_KEY_Y, KeyCode::Y },
//            { GLFW_KEY_Z, KeyCode::Z },
//
//            { GLFW_KEY_LEFT_BRACKET, KeyCode::LeftBracket },  /* [ */
//            { GLFW_KEY_BACKSLASH, KeyCode::Backslash },  /* \ */
//            { GLFW_KEY_RIGHT_BRACKET, KeyCode::RightBracket },  /* ] */
//            { GLFW_KEY_GRAVE_ACCENT, KeyCode::GraveAccent },  /* ` */
//            { GLFW_KEY_WORLD_1, KeyCode::World1 },  /* non-US #1 */
//            { GLFW_KEY_WORLD_2, KeyCode::World2 },  /* non-US #2 */
//
//
//            /* Function keys */
//            { GLFW_KEY_ESCAPE, KeyCode::Escape },
//            { GLFW_KEY_ENTER, KeyCode::Enter },
//            { GLFW_KEY_TAB, KeyCode::Tab },
//            { GLFW_KEY_BACKSPACE, KeyCode::Backspace },
//            { GLFW_KEY_INSERT, KeyCode::Insert },
//            { GLFW_KEY_DELETE, KeyCode::Delete },
//            { GLFW_KEY_RIGHT, KeyCode::Right },
//            { GLFW_KEY_LEFT, KeyCode::Left },
//            { GLFW_KEY_DOWN, KeyCode::Down },
//            { GLFW_KEY_UP, KeyCode::Up },
//            { GLFW_KEY_PAGE_UP, KeyCode::PageUp },
//            { GLFW_KEY_PAGE_DOWN, KeyCode::PageDown},
//            { GLFW_KEY_HOME, KeyCode::Home },
//            { GLFW_KEY_END, KeyCode::End},
//            { GLFW_KEY_CAPS_LOCK, KeyCode::CapsLock },
//            { GLFW_KEY_SCROLL_LOCK, KeyCode::ScrollLock},
//            { GLFW_KEY_NUM_LOCK, KeyCode::NumLock},
//            { GLFW_KEY_PRINT_SCREEN, KeyCode::PrintScreen},
//            { GLFW_KEY_PAUSE, KeyCode::Pause},
//
//            { GLFW_KEY_F1, KeyCode::F1 },
//            { GLFW_KEY_F2, KeyCode::F2 },
//            { GLFW_KEY_F3, KeyCode::F3 },
//            { GLFW_KEY_F4, KeyCode::F4 },
//            { GLFW_KEY_F5, KeyCode::F5 },
//            { GLFW_KEY_F6, KeyCode::F6 },
//            { GLFW_KEY_F7, KeyCode::F7 },
//            { GLFW_KEY_F8, KeyCode::F8 },
//            { GLFW_KEY_F9, KeyCode::F9 },
//            { GLFW_KEY_F10, KeyCode::F10 },
//            { GLFW_KEY_F11, KeyCode::F11 },
//            { GLFW_KEY_F12, KeyCode::F12 },
//            { GLFW_KEY_F13, KeyCode::F13 },
//            { GLFW_KEY_F14, KeyCode::F14 },
//            { GLFW_KEY_F15, KeyCode::F15 },
//            { GLFW_KEY_F16, KeyCode::F16 },
//            { GLFW_KEY_F17, KeyCode::F17 },
//            { GLFW_KEY_F18, KeyCode::F18 },
//            { GLFW_KEY_F19, KeyCode::F19 },
//            { GLFW_KEY_F20, KeyCode::F20 },
//            { GLFW_KEY_F21, KeyCode::F21 },
//            { GLFW_KEY_F22, KeyCode::F22 },
//            { GLFW_KEY_F23, KeyCode::F23 },
//            { GLFW_KEY_F24, KeyCode::F24 },
//            { GLFW_KEY_F25, KeyCode::F25 },
//
//            { GLFW_KEY_KP_0, KeyCode::KP0 },
//            { GLFW_KEY_KP_1, KeyCode::KP1 },
//            { GLFW_KEY_KP_2, KeyCode::KP2 },
//            { GLFW_KEY_KP_3, KeyCode::KP3 },
//            { GLFW_KEY_KP_4, KeyCode::KP4 },
//            { GLFW_KEY_KP_5, KeyCode::KP5 },
//            { GLFW_KEY_KP_6, KeyCode::KP6 },
//            { GLFW_KEY_KP_7, KeyCode::KP7 },
//            { GLFW_KEY_KP_8, KeyCode::KP8 },
//            { GLFW_KEY_KP_9, KeyCode::KP9 },
//            { GLFW_KEY_KP_DECIMAL, KeyCode::KPDecimal },
//            { GLFW_KEY_KP_DIVIDE, KeyCode::KPDivide },
//            { GLFW_KEY_KP_MULTIPLY, KeyCode::KPMultiply },
//            { GLFW_KEY_KP_SUBTRACT, KeyCode::KPSubtract },
//            { GLFW_KEY_KP_ADD, KeyCode::KPAdd },
//            { GLFW_KEY_KP_ENTER, KeyCode::KPEnter },
//            { GLFW_KEY_KP_EQUAL, KeyCode::KPEqual },
//
//            { GLFW_KEY_LEFT_SHIFT, KeyCode::LeftShift },
//            { GLFW_KEY_LEFT_CONTROL, KeyCode::LeftControl },
//            { GLFW_KEY_LEFT_ALT, KeyCode::LeftAlt },
//            { GLFW_KEY_LEFT_SUPER, KeyCode::LeftSuper },
//            { GLFW_KEY_RIGHT_SHIFT, KeyCode::RightShift },
//            { GLFW_KEY_RIGHT_CONTROL, KeyCode::RightControl },
//            { GLFW_KEY_RIGHT_ALT, KeyCode::RightAlt },
//            { GLFW_KEY_RIGHT_SUPER, KeyCode::RightSuper },
//
//            { GLFW_KEY_MENU, KeyCode::Menu },
//            { GLFW_KEY_LAST, KeyCode::KeyLast },
//    };
//
//    return s_key_map[glfwKey];
//}
//
//
//static MouseKeyCode GLFWToGobotMouseKey(uint16_t glfwKey)
//{
//
//    static std::map<uint16_t, MouseKeyCode> s_key_map = {
//            { GLFW_MOUSE_BUTTON_1, MouseKeyCode::Button1 },
//            { GLFW_MOUSE_BUTTON_2, MouseKeyCode::Button2 },
//            { GLFW_MOUSE_BUTTON_3, MouseKeyCode::Button3 },
//            { GLFW_MOUSE_BUTTON_4, MouseKeyCode::Button4 },
//            { GLFW_MOUSE_BUTTON_5, MouseKeyCode::Button5 },
//            { GLFW_MOUSE_BUTTON_6, MouseKeyCode::Button6 },
//            { GLFW_MOUSE_BUTTON_7, MouseKeyCode::Button7 },
//            { GLFW_MOUSE_BUTTON_8, MouseKeyCode::Button8 },
//            { GLFW_MOUSE_BUTTON_LEFT, MouseKeyCode::ButtonLeft },
//            { GLFW_MOUSE_BUTTON_RIGHT, MouseKeyCode::ButtonRight },
//            { GLFW_MOUSE_BUTTON_MIDDLE, MouseKeyCode::ButtonMiddle },
//            { GLFW_MOUSE_BUTTON_LAST, MouseKeyCode::ButtonLast }
//    };
//
//    return s_key_map[glfwKey];
//}

}