/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
*/

#include "gobot/core/io/image_loader_stb.hpp"

#include "gobot/error_macros.hpp"

#include <stb_image.h>

namespace gobot {

ImageLoaderStb::ImageLoaderStb() {
    Image::s_png_mem_loader_func = LoadMemImage;
    Image::s_jpg_mem_loader_func = LoadMemImage;
}

Ref<Image> ImageLoaderStb::LoadImage(const std::vector<uint8_t>& byte_array, LoaderFlags flags, float scale) {
    (void) flags;
    (void) scale;
    return DecodeImage(byte_array.data(), static_cast<int>(byte_array.size()));
}

void ImageLoaderStb::GetRecognizedExtensions(std::vector<std::string>* extensions) const {
    extensions->push_back("png");
    extensions->push_back("jpg");
    extensions->push_back("jpeg");
    extensions->push_back("bmp");
    extensions->push_back("tga");
    extensions->push_back("psd");
    extensions->push_back("gif");
    extensions->push_back("hdr");
    extensions->push_back("pic");
    extensions->push_back("pnm");
}

Ref<Image> ImageLoaderStb::LoadMemImage(const uint8_t* source, int size) {
    return DecodeImage(source, size);
}

Ref<Image> ImageLoaderStb::DecodeImage(const uint8_t* source, int size) {
    ERR_FAIL_COND_V(!source, nullptr);
    ERR_FAIL_COND_V(size <= 0, nullptr);

    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* pixels = stbi_load_from_memory(source, size, &width, &height, &channels, 0);
    ERR_FAIL_COND_V_MSG(!pixels, nullptr, stbi_failure_reason());

    ImageFormat format;
    switch (channels) {
        case STBI_grey:
            format = ImageFormat::L8;
            break;
        case STBI_grey_alpha:
            format = ImageFormat::LA8;
            break;
        case STBI_rgb:
            format = ImageFormat::RGB8;
            break;
        case STBI_rgb_alpha:
            format = ImageFormat::RGBA8;
            break;
        default:
            stbi_image_free(pixels);
            LOG_ERROR("Unsupported image channel count: {}", channels);
            return nullptr;
    }

    const size_t byte_count = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(channels);
    std::vector<uint8_t> data(pixels, pixels + byte_count);
    stbi_image_free(pixels);
    return MakeRef<Image>(width, height, false, format, data);
}

}
