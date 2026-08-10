/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include <gobot/physics/ipc_batch_solver.hpp>
#include <gobot/physics/ipc_solver.hpp>

namespace {

constexpr std::array<double, 16> kIdentity{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0};

gobot::IpcSceneArtifact MakeArtifact() {
    gobot::IpcSceneArtifact artifact;
    artifact.schema_version = 1;
    artifact.producer = "gobot-test";
    artifact.producer_version = "1";
    artifact.format = "gobot-ipc";
    artifact.manifest = R"({"schema_version":1,"format":"gobot-ipc"})";
    artifact.manifest_sha256 = "test";
    return artifact;
}

gobot::IpcSolverDeviceBufferView MakeDeviceBuffer(
        double* data, std::initializer_list<std::size_t> shape) {
    gobot::IpcSolverDeviceBufferView result;
    result.data = data;
    result.device_index = 0;
    result.scalar_type = gobot::IpcSolverDeviceScalarType::Float64;
    result.rank = static_cast<std::uint32_t>(shape.size());
    std::size_t axis = 0;
    for (const std::size_t value : shape) {
        result.shape[axis++] = value;
    }
    std::size_t stride = 1;
    for (std::size_t index = shape.size(); index-- > 0;) {
        result.stride[index] = stride;
        stride *= result.shape[index];
    }
    return result;
}

} // namespace

TEST(TestIpcSolver, reports_missing_and_incompatible_modules) {
    std::string error;
    EXPECT_FALSE(gobot::IpcSolverSession::IsModuleAvailable(
            "/definitely/not/a/gobot/module.so", &error));
    EXPECT_NE(error.find("Cannot load IPC solver module"), std::string::npos);

    error.clear();
    EXPECT_FALSE(gobot::IpcSolverSession::IsModuleAvailable(
            GOBOT_TEST_IPC_SOLVER_BAD_ABI_PATH, &error));
    EXPECT_NE(error.find("ABI"), std::string::npos);
}

TEST(TestIpcSolver, owns_stable_state_and_forwards_session_operations) {
    std::string error;
    ASSERT_TRUE(gobot::IpcSolverSession::IsModuleAvailable(
            GOBOT_TEST_IPC_SOLVER_MODULE_PATH, &error)) << error;

    auto session = gobot::IpcSolverSession::Create(
            MakeArtifact(), gobot::IpcSolverConfig{},
            GOBOT_TEST_IPC_SOLVER_MODULE_PATH, &error);
    ASSERT_NE(session, nullptr) << error;
    ASSERT_EQ(session->GetDeformableBodies().size(), 1);
    ASSERT_EQ(session->GetAffineBodies().size(), 1);
    EXPECT_EQ(session->GetDeformableBodies()[0].path, "/World/Soft");
    EXPECT_EQ(session->GetDeformableBodies()[0].element_count, 2);
    EXPECT_EQ(session->GetAffineBodies()[0].path, "/World/Robot/Finger");
    EXPECT_EQ(session->GetDiagnostics().provider_name, "fake-ipc");
    EXPECT_TRUE(session->GetDiagnostics().valid);

    const double* positions_storage = session->GetDeformablePositions().data();
    const double* velocities_storage = session->GetDeformableVelocities().data();
    const double* contact_forces_storage =
            session->GetDeformableContactForces().data();
    const double* affine_storage = session->GetAffineTransforms().data();
    ASSERT_TRUE(session->Step(2)) << session->GetLastError();
    EXPECT_EQ(session->GetDiagnostics().frame, 2);
    EXPECT_DOUBLE_EQ(session->GetDeformablePositions()[0], 0.5);
    EXPECT_DOUBLE_EQ(session->GetDeformableVelocities()[0], 0.25);
    EXPECT_DOUBLE_EQ(session->GetDeformableContactForces()[2], 2.0);
    EXPECT_DOUBLE_EQ(session->GetDeformableContactForces()[5], 4.0);
    EXPECT_EQ(session->GetDeformablePositions().data(), positions_storage);
    EXPECT_EQ(session->GetDeformableVelocities().data(), velocities_storage);
    EXPECT_EQ(session->GetDeformableContactForces().data(),
              contact_forces_storage);
    EXPECT_EQ(session->GetAffineTransforms().data(), affine_storage);

    std::array<double, 16> target{
            1.0, 0.0, 0.0, 0.25,
            0.0, 1.0, 0.0, -0.5,
            0.0, 0.0, 1.0, 1.5,
            0.0, 0.0, 0.0, 1.0};
    ASSERT_TRUE(session->SetAffineTarget(
            "/World/Robot/Finger", target.data())) << session->GetLastError();
    ASSERT_TRUE(session->Step()) << session->GetLastError();
    EXPECT_DOUBLE_EQ(session->GetAffineTransforms()[3], 0.25);
    EXPECT_DOUBLE_EQ(session->GetAffineTransforms()[7], -0.5);
    EXPECT_DOUBLE_EQ(session->GetAffineTransforms()[11], 1.5);

    ASSERT_TRUE(session->SetJointTarget(
            "/World/Robot/FingerJoint", 0.75)) << session->GetLastError();
    ASSERT_TRUE(session->Step()) << session->GetLastError();
    EXPECT_DOUBLE_EQ(session->GetAffineTransforms()[3], 0.75);

    EXPECT_FALSE(session->SetAffineTarget("/unknown", target.data()));
    EXPECT_NE(session->GetLastError().find("unknown"), std::string::npos);
    EXPECT_FALSE(session->SetJointTarget("/unknown", 0.0));
    EXPECT_NE(session->GetLastError().find("invalid"), std::string::npos);
    EXPECT_FALSE(session->Step(0));
    EXPECT_NE(session->GetLastError().find("positive"), std::string::npos);

    ASSERT_TRUE(session->Reset()) << session->GetLastError();
    EXPECT_EQ(session->GetDiagnostics().frame, 0);
    EXPECT_DOUBLE_EQ(session->GetDeformablePositions()[0], 0.0);
    EXPECT_DOUBLE_EQ(session->GetDeformablePositions()[2], 1.0);
    EXPECT_DOUBLE_EQ(session->GetAffineTransforms()[3], 0.0);
    EXPECT_DOUBLE_EQ(session->GetDeformableContactForces()[2], 0.0);
    EXPECT_DOUBLE_EQ(session->GetDeformableContactForces()[5], 0.0);
    EXPECT_EQ(session->GetDeformablePositions().data(), positions_storage);
    EXPECT_EQ(session->GetAffineTransforms().data(), affine_storage);
    EXPECT_EQ(session->GetDeformableContactForces().data(),
              contact_forces_storage);
}

TEST(TestIpcSolver, forwards_and_validates_contact_activation_distance) {
    gobot::IpcSolverConfig config;
    EXPECT_DOUBLE_EQ(config.contact_activation_distance, 0.01);
    config.contact_activation_distance = 0.0005;

    std::string error;
    auto session = gobot::IpcSolverSession::Create(
            MakeArtifact(), config, GOBOT_TEST_IPC_SOLVER_MODULE_PATH, &error);
    ASSERT_NE(session, nullptr) << error;
    ASSERT_GE(session->GetDeformablePositions().size(), 2);
    EXPECT_DOUBLE_EQ(session->GetDeformablePositions()[1], 0.0005);

    for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        config.contact_activation_distance = invalid;
        error.clear();
        EXPECT_EQ(gobot::IpcSolverSession::Create(
                          MakeArtifact(), config, GOBOT_TEST_IPC_SOLVER_MODULE_PATH, &error),
                  nullptr);
        EXPECT_NE(error.find("activation distance"), std::string::npos);
    }
}

TEST(TestIpcBatchSolver, validates_extension_abi_and_owns_device_buffer_contract) {
    std::string error;
    EXPECT_FALSE(gobot::IpcBatchSolverSession::IsModuleAvailable(
            GOBOT_TEST_IPC_SOLVER_BAD_ABI_PATH, &error));
    EXPECT_NE(error.find("ABI"), std::string::npos);

    error.clear();
    ASSERT_TRUE(gobot::IpcBatchSolverSession::IsModuleAvailable(
            GOBOT_TEST_IPC_SOLVER_MODULE_PATH, &error)) << error;

    gobot::IpcBatchSolverConfig config;
    config.environment_count = 4;
    config.environments_per_shard = 2;
    auto session = gobot::IpcBatchSolverSession::Create(
            MakeArtifact(), config, GOBOT_TEST_IPC_SOLVER_MODULE_PATH, &error);
    ASSERT_NE(session, nullptr) << error;
    ASSERT_EQ(session->GetDeformableBodies().size(), 1);
    ASSERT_EQ(session->GetAffineBodies().size(), 1);
    EXPECT_EQ(session->GetDiagnostics().provider_name, "fake-ipc-batch");
    EXPECT_EQ(session->GetDiagnostics().environment_count, 4);
    EXPECT_EQ(session->GetDiagnostics().shard_count, 2);
    EXPECT_FALSE(session->Step());
    EXPECT_NE(session->GetLastError().find("not bound"), std::string::npos);

    std::vector<double> positions(4 * 2 * 3);
    std::vector<double> velocities(4 * 2 * 3);
    std::vector<double> contact_forces(4 * 2 * 3);
    std::vector<double> targets(4 * 1 * 4 * 4);
    std::vector<double> transforms(4 * 1 * 4 * 4);
    std::vector<double> wrenches(4 * 1 * 6);
    for (std::size_t environment = 0; environment < 4; ++environment) {
        std::copy_n(kIdentity.begin(), 16,
                    targets.begin() + static_cast<std::ptrdiff_t>(environment * 16));
    }
    gobot::IpcBatchSolverModuleBuffers buffers{
            MakeDeviceBuffer(positions.data(), {4, 2, 3}),
            MakeDeviceBuffer(velocities.data(), {4, 2, 3}),
            MakeDeviceBuffer(contact_forces.data(), {4, 2, 3}),
            MakeDeviceBuffer(targets.data(), {4, 1, 4, 4}),
            MakeDeviceBuffer(transforms.data(), {4, 1, 4, 4}),
            MakeDeviceBuffer(wrenches.data(), {4, 1, 6})};
    ASSERT_TRUE(session->BindDeviceBuffers(buffers))
            << session->GetLastError();
    const double* position_storage = positions.data();
    const double* wrench_storage = wrenches.data();
    ASSERT_TRUE(session->Step(2)) << session->GetLastError();
    EXPECT_EQ(session->GetDiagnostics().frame, 2);
    EXPECT_DOUBLE_EQ(positions[0], 0.5);
    EXPECT_DOUBLE_EQ(velocities[0], 0.25);
    EXPECT_DOUBLE_EQ(contact_forces[2], 2.0);
    EXPECT_DOUBLE_EQ(wrenches[2], 2.0);
    EXPECT_EQ(positions.data(), position_storage);
    EXPECT_EQ(wrenches.data(), wrench_storage);
    ASSERT_TRUE(session->Synchronize()) << session->GetLastError();

    targets[3] = 0.75;
    ASSERT_TRUE(session->Step()) << session->GetLastError();
    EXPECT_DOUBLE_EQ(transforms[3], 0.75);
    ASSERT_TRUE(session->ResetFull()) << session->GetLastError();
    EXPECT_EQ(session->GetDiagnostics().frame, 0);
    EXPECT_DOUBLE_EQ(positions[0], 0.0);
    EXPECT_DOUBLE_EQ(positions[2], 1.0);
    EXPECT_DOUBLE_EQ(wrenches[2], 0.0);

    auto invalid = buffers;
    invalid.affine_contact_wrenches.stride[1] = 7;
    EXPECT_FALSE(session->BindDeviceBuffers(invalid));
    EXPECT_NE(session->GetLastError().find("contiguous"), std::string::npos);
}
