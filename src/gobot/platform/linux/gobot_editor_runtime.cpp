/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 2026-05-24
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/main/main.hpp"
#include "gobot/platform/editor_embedded_resources.hpp"
#include "gobot/platform/linux/os_linux.hpp"

using namespace gobot;

extern "C" int gobot_editor_main(int argc, char* argv[]) {
    (void)GetEmbeddedEditorIconSvgSize();

    if (!Main::Setup(argc, argv)) {
        return -1;
    }

    LinuxOS os;

    if (Main::Start()) {
        os.Run();
    }

    Main::Cleanup();
    return 0;
}
