/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace gobot::editor_detail {

[[nodiscard]] bool IsSceneResourcePath(const std::filesystem::path& path);

void AppendExampleProjectDirectories(const std::filesystem::path& examples_dir,
                                     std::vector<std::string>& projects);

} // namespace gobot::editor_detail
