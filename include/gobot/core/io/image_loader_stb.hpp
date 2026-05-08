/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
*/

#pragma once

#include "gobot/core/io/image_loader.hpp"
#include "gobot/core/types.hpp"

namespace gobot {

class ImageLoaderStb : public ImageFormatLoader {
    GOBCLASS(ImageLoaderStb, ImageFormatLoader)
public:
    ImageLoaderStb();

    Ref<Image> LoadImage(const std::vector<uint8_t>& byte_array, LoaderFlags flags, float scale) override;

    void GetRecognizedExtensions(std::vector<std::string>* extensions) const override;

private:
    static Ref<Image> LoadMemImage(const uint8_t* source, int size);
    static Ref<Image> DecodeImage(const uint8_t* source, int size);
};

}
