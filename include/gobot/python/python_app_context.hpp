/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include "gobot_export.h"

namespace gobot {
class EngineContext;
class HeadlessRenderContext;
class Node;
}

namespace gobot::python {

GOBOT_EXPORT void SetActiveAppContext(EngineContext* context);

GOBOT_EXPORT void RegisterExternalAppContext(EngineContext* context);

GOBOT_EXPORT void UnregisterExternalAppContext(EngineContext* context);

GOBOT_EXPORT EngineContext* GetActiveAppContextOrNull();

GOBOT_EXPORT EngineContext& GetActiveAppContext();

GOBOT_EXPORT bool IsAppContextLive(const EngineContext* context);

GOBOT_EXPORT EngineContext* FindAppContextForSceneRoot(const Node* scene_node);

GOBOT_EXPORT std::shared_ptr<EngineContext> CreateAppContext();

GOBOT_EXPORT HeadlessRenderContext& EnsureHeadlessRenderContext();

GOBOT_EXPORT void ShutdownHeadlessRenderContext();

} // namespace gobot::python
