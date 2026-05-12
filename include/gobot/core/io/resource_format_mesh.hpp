/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/io/resource_loader.hpp"

namespace gobot {

class GOBOT_EXPORT ResourceFormatLoaderMesh : public ResourceFormatLoader {
    GOBCLASS(ResourceFormatLoaderMesh, ResourceFormatLoader)

public:
    Ref<Resource> Load(const std::string& path,
                       const std::string& original_path = "",
                       CacheMode cache_mode = CacheMode::Reuse) override;

    void GetRecognizedExtensionsForType(const std::string& type,
                                        std::vector<std::string>* extensions) const override;

    void GetRecognizedExtensions(std::vector<std::string>* extensions) const override;

    [[nodiscard]] bool HandlesType(const std::string& type) const override;

    [[nodiscard]] bool Exists(const std::string& path) const override;
};

} // namespace gobot
