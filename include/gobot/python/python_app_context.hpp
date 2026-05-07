/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include "gobot_export.h"

namespace gobot {
class EngineContext;
}

namespace gobot::python {

GOBOT_EXPORT void SetActiveAppContext(EngineContext* context);

GOBOT_EXPORT EngineContext* GetActiveAppContextOrNull();

GOBOT_EXPORT EngineContext& GetActiveAppContext();

} // namespace gobot::python
