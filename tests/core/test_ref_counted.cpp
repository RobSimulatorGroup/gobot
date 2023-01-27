/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-27
*/

#include <gtest/gtest.h>

#include <gobot/core/ref_counted.hpp>
#include <gobot/core/types.hpp>
#include <gobot/log.hpp>
#include <gobot/core/registration.hpp>
#include <gobot/core/notification_enum.hpp>

namespace gobot {

class TestResource : public RefCounted {
    GOBCLASS(TestResource, RefCounted)
public:
    TestResource() = default;
};

}

GOBOT_REGISTRATION {

    Class_<TestResource>("TestResource")
        .constructor()(CtorAsRawPtr);
    gobot::Type::register_wrapper_converter_for_base_classes<Ref<TestResource>, Ref<RefCounted>>();
};


TEST(TestRefCounted, test_count) {
    gobot::Ref<gobot::RefCounted> p;
    p = gobot::MakeRef<gobot::TestResource>();
    ASSERT_TRUE(p.UseCount() == 1);

    gobot::Ref<gobot::RefCounted> p1 = p;
    ASSERT_TRUE(p.UseCount() == 2);

    ASSERT_EQ(p.Get(), p1.Get());

    p1.Reset();
    ASSERT_TRUE(p->GetReferenceCount() == 1);

    p.Reset();
    ASSERT_TRUE(p.UseCount() == 0);
}

TEST(TestRefCounted, test_nullptr) {
    gobot::Ref<gobot::RefCounted> p{nullptr};
    ASSERT_TRUE(p.UseCount() == 0);
    ASSERT_FALSE(p.operator bool());

    p = gobot::MakeRef<gobot::TestResource>();
    ASSERT_TRUE(p->Unique());
    ASSERT_TRUE(p.operator bool());
    ASSERT_TRUE(p.IsValid());
}


TEST(TestRefRegister, test_ref) {
    gobot::Ref<gobot::RefCounted> p{nullptr};
    gobot::Variant ref(p);
    ASSERT_TRUE(ref.get_type().is_wrapper());

    ASSERT_TRUE(ref.get_type().get_wrapper_holder_type() == gobot::WrapperHolderType::Ref);

    {
        gobot::Variant resource = gobot::MakeRef<gobot::TestResource>();
        ASSERT_TRUE(resource.can_convert<gobot::Ref<gobot::RefCounted>>());
        p = resource.convert<gobot::Ref<gobot::RefCounted>>();
    }

    ASSERT_TRUE(p.UseCount() == 1);
}


TEST(TestRefRegister, test_get_wrapped_instance) {
    gobot::Variant resource = gobot::MakeRef<gobot::TestResource>();
    gobot::Instance instance = resource;
    ASSERT_TRUE(instance.get_wrapped_instance().get_type() == gobot::Type::get<gobot::TestResource*>());
}

TEST(TestRefRegister, test_create) {
    gobot::Variant resource = gobot::Type::get<gobot::TestResource>().create();
    ASSERT_TRUE(resource.can_convert<gobot::TestResource*>());
    ASSERT_TRUE(resource.get_type().is_derived_from<gobot::TestResource>());
}

TEST(TestRefRegister, test_init_ref_with_ref) {
    auto p = gobot::MakeRef<gobot::TestResource>();

    gobot::Ref<gobot::TestResource> p2(p.Get());
    ASSERT_TRUE(p.UseCount() == 2);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}