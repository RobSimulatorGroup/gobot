/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot_export.h"

#include <memory>
#include <string>

namespace gobot {

class RenderServer;

class GOBOT_EXPORT HeadlessRenderContext {
public:
    HeadlessRenderContext();
    ~HeadlessRenderContext();

    HeadlessRenderContext(const HeadlessRenderContext&) = delete;
    HeadlessRenderContext& operator=(const HeadlessRenderContext&) = delete;

    bool Initialize();
    void Shutdown();
    bool IsReady() const;
    const std::string& GetLastError() const;

private:
    struct PlatformContext;

    std::unique_ptr<PlatformContext> platform_;
    std::unique_ptr<RenderServer> render_server_;
    std::string last_error_;
};

}
