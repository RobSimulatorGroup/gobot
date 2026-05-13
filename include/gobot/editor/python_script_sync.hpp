/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "gobot/core/object.hpp"

namespace gobot {

class Node;

void SyncPythonScriptResourceSource(const std::string& local_path,
                                    const std::string& source_code,
                                    Node* scene_root);

} // namespace gobot
