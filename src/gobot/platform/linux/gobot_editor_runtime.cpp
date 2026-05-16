/*
 * The gobot is a robot simulation platform.
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
