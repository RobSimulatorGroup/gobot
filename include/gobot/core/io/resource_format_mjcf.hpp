/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include "gobot/core/io/resource_loader.hpp"

namespace gobot {

class GOBOT_EXPORT ResourceFormatLoaderMJCF : public ResourceFormatLoader {
    GOBCLASS(ResourceFormatLoaderMJCF, ResourceFormatLoader)

public:
    Ref<Resource> Load(const std::string& path,
                       const std::string& original_path = "",
                       CacheMode cache_mode = CacheMode::Reuse) override;

    [[nodiscard]] bool RecognizePath(const std::string& path,
                                     const std::string& type_hint = "") const override;

    void GetRecognizedExtensionsForType(const std::string& type,
                                        std::vector<std::string>* extensions) const override;

    void GetRecognizedExtensions(std::vector<std::string>* extensions) const override;

    [[nodiscard]] bool HandlesType(const std::string& type) const override;

    [[nodiscard]] static bool IsMuJoCoAvailable();
};

}
