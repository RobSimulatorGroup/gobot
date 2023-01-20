/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-20
*/

#ifdef BUILD_WITH_PYBIND11
#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#endif

#include "binding/reg.hpp"
#include "binding/core/bind_core.hpp"
#include "binding/core/io/bind_resource.hpp"
#include "gobot/core/object.hpp"

#ifdef BUILD_WITH_PYBIND11
PYBIND11_EMBEDDED_MODULE(gobot, m) {
    using namespace gobot;
#else
GOBOT_REGISTRATION {
    void* m = nullptr;
#endif

    BindCore(m);
    BindResource(m);

};
