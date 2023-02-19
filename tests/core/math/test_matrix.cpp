/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Yingnan Wu<wuyingnan@users.noreply.github.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license
 * document, but changing it is not allowed. This version of the GNU Lesser
 * General Public License incorporates the terms and conditions of version 3 of
 * the GNU General Public License. This file is created by Yingnan Wu, 23-2-10
 */

#include <gtest/gtest.h>

#include <gobot/log.hpp>
#include <gobot/core/math/matrix.hpp>
#include <gobot/core/io/variant_serializer.hpp>

TEST(TestMatrix, test_setter_getter) {
  using namespace gobot;
  Matrix3 matrix{Eigen::Matrix3<real_t>::Random()};
  auto data = matrix.GetMatrixData();
  Matrix3 test;
  test.SetMatrixData(data);
  ASSERT_EQ(matrix, test);
}

TEST(TestMatrix, test_registration) {
  using namespace gobot;
  Matrix3 matrix{Eigen::Matrix3<real_t>::Random()};
  auto json = gobot::VariantSerializer::VariantToJson(matrix);

  Variant test_variant((Matrix3()));
  ASSERT_TRUE(gobot::VariantSerializer::JsonToVariant(test_variant, json));
  ASSERT_EQ(test_variant.get_value<Matrix3>(), matrix);
}