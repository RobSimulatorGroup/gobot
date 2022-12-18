/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-11
*/


#include "gobot/core/io/resource_format_scene.hpp"

namespace gobot {

bool ResourceFormatSaverScene::Save(const Ref<Resource> &resource, const String &path, ResourceSaverFlags flags)
{

}

void ResourceFormatSaverScene::GetRecognizedExtensions(const Ref<Resource> &resource,
                                                       std::vector<String>* extensions) const {

}

bool ResourceFormatSaverScene::Recognize(const Ref<Resource> &resource) const {

}

bool ResourceFormatSaverScene::RecognizePath(const Ref<Resource> &resource, const String &path) const {


}

}