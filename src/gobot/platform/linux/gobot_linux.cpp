/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-2-10
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/main/main.hpp"
#include "gobot/platform/editor_embedded_resources.hpp"
#include "gobot/scene/window.hpp"
#include "gobot/platform/linux/os_linux.hpp"

using namespace gobot;

int main(int argc, char *argv[]) {
    (void)GetEmbeddedEditorIconSvgSize();

    if (!Main::Setup(argc, argv)) {
        return -1;
    }

    LinuxOS os;

    if(Main::Start()) {
        os.Run();
    }

    Main::Cleanup();
    return 0;

}
