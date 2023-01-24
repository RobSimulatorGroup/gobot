/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-1-15
*/

#include <gtest/gtest.h>

#include <gobot/core/ref_counted.hpp>
#include <gobot/core/io/resource_format_scene.hpp>
#include <gobot/core/config/project_setting.hpp>
#include <gobot/scene/resources/cylinder_shape_3d.hpp>

static gobot::Ref<gobot::ResourceFormatLoaderScene> resource_loader_scene;
static gobot::Ref<gobot::ResourceFormatSaverScene> resource_saver_scene;

TEST(TestResource, test_save) {
    gobot::ProjectSettings project_settings;

    auto* project_setting = gobot::ProjectSettings::GetSingleton();
    project_setting->SetProjectPath("/tmp/test_project");

    resource_saver_scene = gobot::MakeRef<gobot::ResourceFormatSaverScene>();
    gobot::ResourceSaver::AddResourceFormatSaver(resource_saver_scene, true);


    gobot::Ref<gobot::CylinderShape3D> cy = gobot::MakeRef<gobot::CylinderShape3D>();

    gobot::ResourceSaver::Save(cy, "res://cyl.jres", gobot::ResourceSaverFlags::ReplaceSubResourcePaths);

}