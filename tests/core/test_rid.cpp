/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Zikun Yu, 23-3-20
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "gobot/core/rid.hpp"

TEST(TestRID, constructor) {
    gobot::RID rid;

    ASSERT_EQ(rid.GetID(), 0);
}

TEST(TestRID, factory) {
    auto rid = gobot::RID::FromUint64(1);

    ASSERT_EQ(rid.GetID(), 1);
}

TEST(TestRID, operators) {
    using namespace gobot;

    auto rid = RID::FromUint64(1);

    auto rid_zero = RID::FromUint64(0);
    auto rid_one = RID::FromUint64(1);
    auto rid_two = RID::FromUint64(2);

    ASSERT_FALSE(rid == rid_zero);
    ASSERT_TRUE(rid == rid_one);
    ASSERT_FALSE(rid == rid_two);

    ASSERT_FALSE(rid < rid_zero);
    ASSERT_FALSE(rid < rid_one);
    ASSERT_TRUE(rid < rid_two);

    ASSERT_FALSE(rid <= rid_zero);
    ASSERT_TRUE(rid <= rid_one);
    ASSERT_TRUE(rid <= rid_two);

    ASSERT_TRUE(rid > rid_zero);
    ASSERT_FALSE(rid > rid_one);
    ASSERT_FALSE(rid > rid_two);

    ASSERT_TRUE(rid >= rid_zero);
    ASSERT_TRUE(rid >= rid_one);
    ASSERT_FALSE(rid >= rid_two);

    ASSERT_TRUE(rid != rid_zero);
    ASSERT_FALSE(rid != rid_one);
    ASSERT_TRUE(rid != rid_two);
}

TEST(TestRID, methods) {
    using namespace gobot;

    auto rid_zero = RID::FromUint64(0);
    auto rid_one = RID::FromUint64(1);

    ASSERT_FALSE(rid_zero.IsValid());
    ASSERT_TRUE(rid_zero.IsNull());

    ASSERT_TRUE(rid_one.IsValid());
    ASSERT_FALSE(rid_one.IsNull());

    ASSERT_EQ(rid_one.GetLocalIndex(), 1);

    ASSERT_EQ(RID::FromUint64(4'294'967'295).GetLocalIndex(), 4'294'967'295);
    ASSERT_EQ(RID::FromUint64(4'294'967'297).GetLocalIndex(), 1);
}

