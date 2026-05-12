/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/io/resource.hpp"
#include "gobot/core/io/resource_loader.hpp"

namespace gobot {

class GOBOT_EXPORT PythonScript : public Resource {
    GOBCLASS(PythonScript, Resource)

public:
    void SetSourceCode(const std::string& source_code);

    const std::string& GetSourceCode() const;

private:
    std::string source_code_;
};

class GOBOT_EXPORT ResourceFormatLoaderPythonScript : public ResourceFormatLoader {
    GOBCLASS(ResourceFormatLoaderPythonScript, ResourceFormatLoader)

public:
    Ref<Resource> Load(const std::string& path,
                       const std::string& original_path = "",
                       CacheMode cache_mode = CacheMode::Reuse) override;

    void GetRecognizedExtensions(std::vector<std::string>* extensions) const override;

    [[nodiscard]] bool HandlesType(const std::string& type) const override;
};

} // namespace gobot
