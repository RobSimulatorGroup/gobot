/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-1-15
*/

#include <gtest/gtest.h>


#include <gobot/core/io/resource.hpp>
#include <gobot/core/io/variant_serializer.hpp>
#include <gobot/core/types.hpp>
#include <gobot/log.hpp>


TEST(TestVariantSerializer, test_vector_int) {
    std::vector<int> vector_int{1, 2, 3};
    gobot::Variant var(vector_int);
    auto json = gobot::VariantSerializer::VariantToJson(var);
    LOG_ERROR("{}", json.dump(4));
    gobot::Variant aa = gobot::VariantSerializer::JsonToVariant(var.get_type(), json);
    LOG_INFO("{}", aa.get_type().get_name().data());

}