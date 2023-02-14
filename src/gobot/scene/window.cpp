/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-1-14
*/


#include "gobot/scene/window.hpp"
#include "gobot/drivers/sdl/sdl_window.hpp"
#include "gobot/core/os/input.hpp"
#include "gobot/graphics/RHI/graphics_context.hpp"
#include "gobot/drivers/sdl/sdl_keycodes.hpp"

#include "gobot/log.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

Window::Window() {
    switch (window_driver_) {
        case WindowDriver::SDL:
        case WindowDriver::Win32:
        case WindowDriver::IOSWindow:
            window_interface_ = std::make_unique<SDLWindow>();
    }

}

Window::~Window() {

}



}