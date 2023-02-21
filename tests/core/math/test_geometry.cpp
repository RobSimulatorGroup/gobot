/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Yingnan Wu<wuyingnan@users.noreply.github.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license
 * document, but changing it is not allowed. This version of the GNU Lesser
 * General Public License incorporates the terms and conditions of version 3 of
 * the GNU General Public License. This file is created by Yingnan Wu, 23-2-10
 */

#include <gtest/gtest.h>

#include <gobot/log.hpp>
#include <gobot/core/math/geometry.hpp>
#include <gobot/core/io/variant_serializer.hpp>

TEST(TestGeoMetry, test_setter_getter) {
    using namespace gobot;
    Isometry3 isometry = Isometry3::Identity();
    auto data = isometry.GetMatrixData();
    Isometry3 test;
    test.SetMatrixData(data);
    ASSERT_EQ(isometry.matrix(), test.matrix());
}

