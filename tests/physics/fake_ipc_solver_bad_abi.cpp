/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/physics/ipc_solver_module_api.hpp"
#include "gobot/physics/ipc_batch_solver_module_api.hpp"

namespace {

const gobot::IpcSolverModuleApi kApi{
        gobot::GOBOT_IPC_SOLVER_MODULE_ABI_VERSION + 1,
        "bad-abi"};

const gobot::IpcBatchSolverModuleApi kBatchApi{
        gobot::GOBOT_IPC_BATCH_SOLVER_MODULE_ABI_VERSION - 1,
        "legacy-batch-abi-v1"};

} // namespace

extern "C" __attribute__((visibility("default")))
const gobot::IpcSolverModuleApi* gobot_ipc_solver_get_api() {
    return &kApi;
}

extern "C" __attribute__((visibility("default")))
const gobot::IpcBatchSolverModuleApi* gobot_ipc_solver_get_batch_api() {
    return &kBatchApi;
}
