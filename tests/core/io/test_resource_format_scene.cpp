/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-1-15
*/

#include <gtest/gtest.h>

#include <gobot/core/ref_counted.hpp>
#include "gobot/core/io/resource_format_scene.hpp"

static gobot::Ref<gobot::ResourceFormatLoaderScene> resource_loader_scene;
static gobot::Ref<gobot::ResourceFormatSaverScene> resource_saver_scene;

TEST(TestResource, test_cast) {
    resource_saver_scene = gobot::MakeRef<gobot::ResourceFormatSaverScene>();
    gobot::ResourceSaver::AddResourceFormatSaver(resource_saver_scene, true);

    resource_loader_scene = gobot::MakeRef<gobot::ResourceFormatLoaderScene>();
    gobot::ResourceLoader::AddResourceFormatLoader(resource_loader_scene, true);


}