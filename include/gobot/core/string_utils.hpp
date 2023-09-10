/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-14
*/

#pragma once

#include <gobot_export.h>
#include <string>
#include <vector>

namespace gobot {

enum class StringSplitBehavior {
    KeepEmptyParts,
    SkipEmptyParts
};

std::vector<std::string> Split(std::string s, std::string delimiter,
                               StringSplitBehavior split_behavior = StringSplitBehavior::KeepEmptyParts);

std::string GOBOT_EXPORT ToLower(std::string_view str);

std::string GOBOT_EXPORT ToUpper(std::string_view str);

std::string GOBOT_EXPORT ReplaceAll(std::string str, const std::string& from, const std::string& to);

std::string GOBOT_EXPORT GetFileExtension(std::string_view path);

bool GOBOT_EXPORT IsNetworkSharePath(std::string_view path);

bool GOBOT_EXPORT IsAbsolutePath(std::string_view path);

bool GOBOT_EXPORT IsRelativePath(std::string_view path);

std::string GOBOT_EXPORT SimplifyPath(std::string_view path);

std::string GOBOT_EXPORT ValidateLocalPath(const std::string& path);

std::string GOBOT_EXPORT GetBaseDir(std::string_view path);

std::string GOBOT_EXPORT PathJoin(std::string_view base, std::string_view file);

}