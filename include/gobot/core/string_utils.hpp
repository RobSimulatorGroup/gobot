/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-14
*/

#pragma once

#include <QFileInfo>
#include <gobot_export.h>

#include "gobot/core/types.hpp"

namespace gobot {

String GOBOT_EXPORT GetFileExtension(const String& path);

bool GOBOT_EXPORT IsNetworkSharePath(const String& path);

bool GOBOT_EXPORT IsAbsolutePath(const String& path);

bool GOBOT_EXPORT IsRelativePath(const String& path);

String GOBOT_EXPORT SimplifyPath(const String& path);

String GOBOT_EXPORT ValidateLocalPath(const String& path);

String GOBOT_EXPORT GetBaseDir(const String& path);

String GOBOT_EXPORT PathJoin(const String &base, const String &file);

// hidden file is the file that name start with "."
bool GOBOT_EXPORT IsHiddenFile(const String& path);

}