/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gobot/core/macros.hpp"

namespace gobot {

class Node;

struct IpcSceneArtifactBlob {
    std::string id;
    std::string encoding;
    std::string sha256;
    std::vector<std::uint8_t> data;
};

struct IpcSceneArtifact {
    std::uint32_t schema_version{0};
    std::string producer;
    std::string producer_version;
    std::string format;
    std::string manifest;
    std::string manifest_sha256;
    std::vector<IpcSceneArtifactBlob> blobs;
};

class GOBOT_EXPORT IpcSceneCompiler {
public:
    static bool Compile(const Node* scene_root,
                        IpcSceneArtifact* artifact,
                        std::string* error = nullptr);
};

} // namespace gobot
