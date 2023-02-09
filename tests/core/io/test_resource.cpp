/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-11
*/

#include <gtest/gtest.h>

#include "gobot/core/ref_counted.hpp"
#include "gobot/core/io/resource.hpp"
#include "gobot/core/types.hpp"
#include "gobot/scene/resources/primitive_mesh.hpp"
#include "gobot/log.hpp"

TEST(TestResource, test_cast) {
    gobot::Ref<gobot::Resource> res(gobot::MakeRef<gobot::Resource>());
    gobot::Instance instance(res);
    gobot::Variant var(res);
    gobot::Instance instance2(var);

    ASSERT_TRUE(instance.get_wrapped_instance().try_convert<gobot::Resource>() != nullptr);
    ASSERT_TRUE(instance2.get_wrapped_instance().try_convert<gobot::Resource>() != nullptr);

    gobot::Ref<gobot::BoxMesh> box_mesh(gobot::MakeRef<gobot::BoxMesh>());
    gobot::Instance instance_box(box_mesh);
    ASSERT_TRUE(instance_box.get_wrapped_instance().try_convert<gobot::Resource>() != nullptr);
}

TEST(TestResource, test_clone) {
    auto box_mesh = gobot::MakeRef<gobot::BoxMesh>();
    box_mesh->SetWidth(10.0);
    auto copy_box = gobot::dynamic_pointer_cast<gobot::BoxMesh>(box_mesh->Clone());
    ASSERT_TRUE(copy_box->GetWidth() == box_mesh->GetWidth());
}

TEST(TestResource, test_resource_cache) {
    auto box_mesh = gobot::MakeRef<gobot::BoxMesh>();
    box_mesh->SetPath("res://box_mesh.jres");
    ASSERT_TRUE(gobot::ResourceCache::GetRef("res://box_mesh.jres").Get() == box_mesh.Get());
}

TEST(TestResource, test_copy_from) {
    auto box_mesh = gobot::MakeRef<gobot::BoxMesh>();
    auto box_mesh2 = gobot::MakeRef<gobot::BoxMesh>();
    box_mesh2->SetWidth(10.0);

    box_mesh->CopyFrom(box_mesh2);
    ASSERT_TRUE(box_mesh->GetWidth() == box_mesh2->GetWidth());
}

TEST(TestResource, test_generate_unique_id) {
    for(int i= 0 ; i< 5; i++) {
        LOG_INFO("{}", gobot::Resource::GenerateResourceUniqueId());
    }
}