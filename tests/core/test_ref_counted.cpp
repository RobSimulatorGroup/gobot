/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-11-27
*/

#include <gtest/gtest.h>

#include <gobot/core/ref_counted.hpp>

namespace {

class TestResource : public gobot::RefCounted {
public:
    TestResource() = default;
};

}

TEST(TestRefCounted, test_count) {
    gobot::Ref<gobot::RefCounted> p;
    gobot::RefWeak<gobot::RefCounted> wp;
    p = godot::make_intrusive<TestResource>();
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