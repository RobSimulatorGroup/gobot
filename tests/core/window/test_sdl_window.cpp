/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-9
*/

#include <gtest/gtest.h>
#include <gobot/drivers/sdl/sdl_window.hpp>
#include <gobot/core/config/project_setting.hpp>
#include <gobot/core/io/image_load.hpp>
#include <gobot/log.hpp>


TEST(TestSDLWindow, test_create) {
    gobot::ProjectSettings project_settings;
    auto ref = gobot::MakeRef<gobot::ResourceFormatLoaderSDLImage>();

    auto image = gobot::Image::LoadFromFile("icon.svg");
    auto* sdl_window = new gobot::SDLWindow();

    sdl_window->SetIcon(image);
    sdl_window->SetEventCallback([](gobot::Event& event) -> void {
        LOG_INFO("{}", event.ToString());
    });
    sdl_window->ProcessEvents();

//    sleep(10);
}