/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
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
