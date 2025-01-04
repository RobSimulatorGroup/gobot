/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-17
*/

#include "gobot/core/io/image_loader_png.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/core/macros.hpp"

#include "png.h"

namespace gobot {

ImageLoaderPNG::ImageLoaderPNG() {
    Image::s_png_mem_loader_func = LoadMemPng;
}

Ref<Image> ImageLoaderPNG::LoadImage(const std::vector<uint8_t>& byte_array, LoaderFlags flags, float scale) {
    std::vector<uint8_t> file_buffer(byte_array.begin(), byte_array.end());
    USING_ENUM_BITWISE_OPERATORS;
    return PngToImage(file_buffer.data(), byte_array.size(), (bool)(flags & LoaderFlags::ForceLinear));
}

void ImageLoaderPNG::GetRecognizedExtensions(std::vector<std::string>* extensions) const {
    extensions->push_back("png");
}

Ref<Image> ImageLoaderPNG::LoadMemPng(const uint8_t* png, int size) {
    // the value of p_force_linear does not matter since it only applies to 16 bit
    auto image = PngToImage(png, size, false);
    ERR_FAIL_COND_V(image.IsValid(), Ref<Image>());

    return image;
}

static bool check_error(const png_image &image) {
    const png_uint_32 failed = PNG_IMAGE_FAILED(image);
    if (failed & PNG_IMAGE_ERROR) {
        return true;
    } else if (failed) {
        LOG_WARN(image.message);
    }
    return false;
}

Ref<Image> ImageLoaderPNG::PngToImage(const uint8_t* source, size_t size, bool force_linear) {
    png_image png_img;
    memset(&png_img, 0, sizeof(png_img));
    png_img.version = PNG_IMAGE_VERSION;

    // fetch image properties
    int success = png_image_begin_read_from_memory(&png_img, source, size);
    ERR_FAIL_COND_V_MSG(check_error(png_img), nullptr, png_img.message);
    ERR_FAIL_COND_V(!success, nullptr);

    // flags to be masked out of input format to give target format
    const png_uint_32 format_mask = ~(
            // convert component order to RGBA
            PNG_FORMAT_FLAG_BGR | PNG_FORMAT_FLAG_AFIRST
            // convert 16 bit components to 8 bit
            | PNG_FORMAT_FLAG_LINEAR
            // convert indexed image to direct color
            | PNG_FORMAT_FLAG_COLORMAP);

    png_img.format &= format_mask;

    ImageFormat dest_format;
    switch (png_img.format) {
        case PNG_FORMAT_GRAY:
            dest_format = ImageFormat::L8;
            break;
        case PNG_FORMAT_GA:
            dest_format = ImageFormat::LA8;
            break;
        case PNG_FORMAT_RGB:
            dest_format = ImageFormat::RGB8;
            break;
        case PNG_FORMAT_RGBA:
            dest_format = ImageFormat::RGBA8;
            break;
        default:
            png_image_free(&png_img); // only required when we return before finish_read
            LOG_ERROR("Unsupported png format.");
            return nullptr;
    }

    if (!force_linear) {
        // assume 16 bit pngs without sRGB or gAMA chunks are in sRGB format
        png_img.flags |= PNG_IMAGE_FLAG_16BIT_sRGB;
    }

    const png_uint_32 stride = PNG_IMAGE_ROW_STRIDE(png_img);
    std::vector<uint8_t> buffer;
    buffer.resize(PNG_IMAGE_BUFFER_SIZE(png_img, stride));
    uint8_t *writer = buffer.data();

    // read image data to buffer and release libpng resources
    success = png_image_finish_read(&png_img, nullptr, writer, stride, nullptr);
    ERR_FAIL_COND_V_MSG(check_error(png_img), nullptr, png_img.message);
    ERR_FAIL_COND_V(!success, nullptr);

    return MakeRef<Image>(png_img.width, png_img.height, false, dest_format, buffer);
}

}
