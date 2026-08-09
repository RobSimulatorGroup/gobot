/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/physics/ipc_solver_module_api.hpp"

namespace {

const gobot::IpcSolverModuleApi kApi{
        gobot::GOBOT_IPC_SOLVER_MODULE_ABI_VERSION + 1,
        "bad-abi"};

} // namespace

extern "C" __attribute__((visibility("default")))
const gobot::IpcSolverModuleApi* gobot_ipc_solver_get_api() {
    return &kApi;
}
