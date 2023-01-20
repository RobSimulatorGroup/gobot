/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-27
*/

#include <gtest/gtest.h>

#include <pybind11/embed.h>
namespace py = pybind11;

TEST(TestPybind, test_pybind_setup) {
    py::scoped_interpreter guard{};

    ASSERT_NO_THROW(py::module_::import("sys"));
}


TEST(TestPybind, test_bind_gobot) {
    py::scoped_interpreter guard{};

    auto gobot = py::module_::import("gobot");


}
