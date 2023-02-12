/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Yingnan Wu<wuyingnan@users.noreply.github.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license
 * document, but changing it is not allowed. This version of the GNU Lesser
 * General Public License incorporates the terms and conditions of version 3 of
 * the GNU General Public License. This file is created by Yingnan Wu, 23-2-10
 */

#include <gtest/gtest.h>

#include <gobot/core/math/matrix.hpp>

TEST(TestMatrix, test_constructor) { 
   gobot::Matrix3 a;
    std::cerr << "\n------------------------------------------" << std::endl;
    if (__cplusplus == 202002L)
      std::cerr << "C++20\n";
    else if (__cplusplus == 201703L)
      std::cerr << "C++17\n";
    else if (__cplusplus == 201402L)
      std::cerr << "C++14\n";
    else if (__cplusplus == 201103L)
      std::cerr << "C++11\n";
    else if (__cplusplus == 199711L)
      std::cerr << "C++98\n";
    else
      std::cerr << "pre-standard C++\n";
}
