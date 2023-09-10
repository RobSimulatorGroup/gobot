/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-17
*/

#pragma once

#include "gobot/core/io/image_loader.hpp"
#include "gobot/core/types.hpp"

namespace gobot {

class ImageLoaderPNG : public ImageFormatLoader {
    GOBCLASS(ImageLoaderPNG, ImageFormatLoader)
public:
    ImageLoaderPNG();

    virtual Ref<Image> LoadImage(const std::vector<uint8_t>& byte_array, LoaderFlags flags, float scale);

    virtual void GetRecognizedExtensions(std::vector<std::string>* extensions) const;

private:
    static Ref<Image> LoadMemPng(const uint8_t* png, int size);

    // Attempt to load png from buffer (p_source, p_size) into p_image
    static Ref<Image> PngToImage(const uint8_t* source, size_t size, bool force_linear);
};

}
