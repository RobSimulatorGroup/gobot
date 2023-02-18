/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-17
*/

#include "gobot/scene/scene_initializer.hpp"
#include "gobot/core/io/resource_format_scene.hpp"

namespace gobot {

static Ref<ResourceFormatSaverScene> s_resource_saver_scene = nullptr;
static Ref<ResourceFormatLoaderScene> s_resource_loader_scene = nullptr;


void SceneInitializer::Init() {
    s_resource_saver_scene = MakeRef<ResourceFormatSaverScene>();
    ResourceSaver::AddResourceFormatSaver(s_resource_saver_scene, true);

    s_resource_loader_scene = MakeRef<ResourceFormatLoaderScene>();
    ResourceLoader::AddResourceFormatLoader(s_resource_loader_scene, true);
}

void SceneInitializer::Destroy() {
    ResourceSaver::RemoveResourceFormatSaver(s_resource_saver_scene);
    s_resource_saver_scene.Reset();

    ResourceLoader::RemoveResourceFormatLoader(s_resource_loader_scene);
    s_resource_loader_scene.Reset();
}

}
