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

namespace {

class TestResource : public gobot::RefCounted {
public:
    TestResource() = default;
};

}

GOBOT_REGISTRATION {

    Class_<TestResource>("TestResource");
    gobot::Type::register_wrapper_converter_for_base_classes<gobot::Ref<TestResource>, gobot::Ref<RefCounted>>();
};


TEST(TestRefCounted, test_count) {
    gobot::Ref<gobot::RefCounted> p;
    gobot::RefWeak<gobot::RefCounted> wp;
    p = gobot::MakeRef<TestResource>();
    ASSERT_TRUE(p.use_count() == 1);
    gobot::Ref<gobot::RefCounted> p1 = p;
    ASSERT_TRUE(p.use_count() == 2);
    ASSERT_TRUE(p.weak_count() == 0);

    wp = p;
    ASSERT_TRUE(p.weak_count() == 1);
    gobot::Ref<gobot::RefCounted> p2 = wp.lock();
    ASSERT_TRUE(p.use_count() == 3);
    ASSERT_EQ(p2.get(), p.get());
    ASSERT_EQ(p2.get(), p1.get());

    wp.reset();
    ASSERT_TRUE(p.weak_count() == 0);
    p2.reset();
    ASSERT_TRUE(p.use_count() == 2);

    p1.reset();
    ASSERT_TRUE(p.use_count() == 1);

    p.reset();
    ASSERT_TRUE(p.use_count() == 0);

}

TEST(TestRefCounted, test_nullptr) {
    gobot::Ref<gobot::RefCounted> p{nullptr};
    ASSERT_TRUE(p.use_count() == 0);
    ASSERT_FALSE(p.operator bool());

    p = gobot::MakeRef<TestResource>();
    ASSERT_TRUE(p.unique());
    ASSERT_TRUE(p.operator bool());
    ASSERT_TRUE(p.is_valid());
}


TEST(TestRefRegister, test_ref) {
    gobot::Ref<gobot::RefCounted> p{nullptr};
    gobot::Variant ref(p);
    ASSERT_TRUE(ref.get_type().is_wrapper());

    ASSERT_TRUE(ref.get_type().get_wrapper_holder_type() == gobot::WrapperHolderType::Ref);

    {
        gobot::Variant resource = gobot::MakeRef<TestResource>();
        ASSERT_TRUE(resource.can_convert<gobot::Ref<gobot::RefCounted>>());
        p = resource.convert<gobot::Ref<gobot::RefCounted>>();
    }

    ASSERT_TRUE(p.use_count() == 1);
}