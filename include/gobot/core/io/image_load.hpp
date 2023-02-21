/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-13
*/

#pragma once

#include "gobot/core/ref_counted.hpp"
#include "gobot/core/io/image.hpp"
#include "gobot/core/io/resource_loader.hpp"

namespace gobot {


class GOBOT_EXPORT ResourceFormatLoaderSDLImage : public ResourceFormatLoader {
    GOBCLASS(ResourceFormatLoaderSDLImage, ResourceFormatLoader)
public:
    ResourceFormatLoaderSDLImage();

    static ResourceFormatLoaderSDLImage* GetInstance();

    Ref<Resource> Load(const String &path, CacheMode cache_mode = CacheMode::Reuse) override;

    void GetRecognizedExtensions(std::vector<String> *extensions) const override;

    [[nodiscard]] bool HandlesType(const String& type) const override;

private:
    static ResourceFormatLoaderSDLImage* s_singleton;
};


}
