/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-10
*/


#include "gobot/core/io/image.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/core/math/math_util.hpp"

#include <stb_image.h>
#include <stb_image_write.h>
#include <magic_enum/magic_enum.hpp>

namespace gobot {

const char *Image::s_format_names[(unsigned int)ImageFormat::MAX] = {
        "Lum8", //luminance
        "LumAlpha8", //luminance-alpha
        "Red8",
        "RedGreen",
        "RGB8",
        "RGBA8",
        "RGBA4444",
        "RGBA5551",
        "RFloat", //float
        "RGFloat",
        "RGBFloat",
        "RGBAFloat",
        "RHalf", //half float
        "RGHalf",
        "RGBHalf",
        "RGBAHalf",
        "RGBE9995",
        "DXT1 RGB8", //s3tc
        "DXT3 RGBA8",
        "DXT5 RGBA8",
        "RGTC Red8",
        "RGTC RedGreen8",
        "BPTC_RGBA",
        "BPTC_RGBF",
        "BPTC_RGBFU",
        "ETC", //etc1
        "ETC2_R11", //etc2
        "ETC2_R11S", //signed", NOT srgb.
        "ETC2_RG11",
        "ETC2_RG11S",
        "ETC2_RGB8",
        "ETC2_RGBA8",
        "ETC2_RGB8A1",
        "ETC2_RA_AS_RG",
        "ImageFormat::DXT5_RA_AS_RG",
        "ASTC_4x4",
        "ASTC_4x4_HDR",
        "ASTC_8x8",
        "ASTC_8x8_HDR",
};

ImageMemLoadFunc Image::s_png_mem_loader_func = nullptr;
ImageMemLoadFunc Image::s_jpg_mem_loader_func = nullptr;

Image::Image()
{
}

Image::Image(int width, int height, bool use_mipmaps, ImageFormat format)
{
    InitializeData(width, height, use_mipmaps, format);
}

Image::Image(int width, int height, bool use_mipmaps, ImageFormat format, const std::vector<uint8_t> &data)
{
    InitializeData(width, height, use_mipmaps, format, data);
}


int Image::GetWidth() const {
    return width_;
}

int Image::GetHeight() const {
    return height_;
}

Vector2i Image::GetSize() const {
    return {width_, height_};
}

bool Image::HasMipmaps() const {
    return use_mipmaps_;
}

int Image::GetMipmapCount() const {
    if (use_mipmaps_) {
        return GetImageRequiredMipmaps(width_, height_, format_);
    } else {
        return 0;
    }
}

int Image::GetImageRequiredMipmaps(int width, int height, ImageFormat format) {
    int mm;
    GetDstImageSize(width, height, format, mm, -1);
    return mm;
}

ImageFormat Image::GetFormat() const {
    return format_;
}

int Image::GetMipmapByteSize(int mipmap) const {
    ERR_FAIL_INDEX_V(mipmap, GetMipmapCount() + 1, -1);

    int ofs, w, h;
    GetMipmapOffsetAndSize(mipmap, ofs, w, h);
    int ofs2;
    GetMipmapOffsetAndSize(mipmap + 1, ofs2, w, h);
    return ofs2 - ofs;
}

int Image::GetMipmapOffset(int mipmap_index) const {
    ERR_FAIL_INDEX_V(mipmap_index, GetMipmapCount() + 1, -1);

    int ofs, w, h;
    GetMipmapOffsetAndSize(mipmap_index, ofs, w, h);
    return ofs;
}

void Image::GetMipmapOffsetAndSize(int mipmap_index, int &r_ofs, int &r_size) const{
    int ofs, w, h;
    GetMipmapOffsetAndSize(mipmap_index, ofs, w, h);
    int ofs2;
    GetMipmapOffsetAndSize(mipmap_index + 1, ofs2, w, h);
    r_ofs = ofs;
    r_size = ofs2 - ofs;
}

void Image::GetMipmapOffsetSizeAndDimensions(int mipmap_index, int &r_ofs, int &r_size, int &w, int &h) const {
    int ofs;
    GetMipmapOffsetAndSize(mipmap_index, ofs, w, h);
    int ofs2, w2, h2;
    GetMipmapOffsetAndSize(mipmap_index + 1, ofs2, w2, h2);
    r_ofs = ofs;
    r_size = ofs2 - ofs;
}

void Image::GetMipmapOffsetAndSize(int mipmap, int &r_offset, int &r_width, int &r_height) const {
    int w = width_;
    int h = height_;
    int ofs = 0;

    int pixel_size = GetFormatPixelSize(format_);
    int pixel_rshift = GetFormatPixelRshift(format_);
    int block = GetFormatBlockSize(format_);
    int minw, minh;
    GetFormatMinPixelSize(format_, minw, minh);

    for (int i = 0; i < mipmap; i++) {
        int bw = w % block != 0 ? w + (block - w % block) : w;
        int bh = h % block != 0 ? h + (block - h % block) : h;

        int s = bw * bh;

        s *= pixel_size;
        s >>= pixel_rshift;
        ofs += s;
        w = std::max(minw, w >> 1);
        h = std::max(minh, h >> 1);
    }

    r_offset = ofs;
    r_width = w;
    r_height = h;
}

Color Image::GetColorAtOfs(const uint8_t *ptr, uint32_t ofs) const {
    switch (format_) {
        case ImageFormat::L8: {
            float l = ptr[ofs] / 255.0;
            return {l, l, l, 1.0};
        }
        case ImageFormat::LA8: {
            float l = ptr[ofs * 2 + 0] / 255.0;
            float a = ptr[ofs * 2 + 1] / 255.0;
            return {l, l, l, a};
        }
        case ImageFormat::R8: {
            float r = ptr[ofs] / 255.0;
            return {r, 0.0, 0.0, 1.0};
        }
        case ImageFormat::RG8: {
            float r = ptr[ofs * 2 + 0] / 255.0;
            float g = ptr[ofs * 2 + 1] / 255.0;
            return {r, g, 0.0, 1.0};
        }
        case ImageFormat::RGB8: {
            float r = ptr[ofs * 3 + 0] / 255.0;
            float g = ptr[ofs * 3 + 1] / 255.0;
            float b = ptr[ofs * 3 + 2] / 255.0;
            return {r, g, b, 1.0};
        }
        case ImageFormat::RGBA8: {
            float r = ptr[ofs * 4 + 0] / 255.0;
            float g = ptr[ofs * 4 + 1] / 255.0;
            float b = ptr[ofs * 4 + 2] / 255.0;
            float a = ptr[ofs * 4 + 3] / 255.0;
            return {r, g, b, a};
        }
        case ImageFormat::RGBA4444: {
            uint16_t u = ((uint16_t *)ptr)[ofs];
            float r = ((u >> 12) & 0xF) / 15.0;
            float g = ((u >> 8) & 0xF) / 15.0;
            float b = ((u >> 4) & 0xF) / 15.0;
            float a = (u & 0xF) / 15.0;
            return {r, g, b, a};
        }
        case ImageFormat::RGB565: {
            uint16_t u = ((uint16_t *)ptr)[ofs];
            float r = (u & 0x1F) / 31.0;
            float g = ((u >> 5) & 0x3F) / 63.0;
            float b = ((u >> 11) & 0x1F) / 31.0;
            return {r, g, b, 1.0};
        }
        case ImageFormat::RF: {
            float r = ((float *)ptr)[ofs];
            return {r, 0.0, 0.0, 1.0};
        }
        case ImageFormat::RGF: {
            float r = ((float *)ptr)[ofs * 2 + 0];
            float g = ((float *)ptr)[ofs * 2 + 1];
            return {r, g, 0.0, 1.0};
        }
        case ImageFormat::RGBF: {
            float r = ((float *)ptr)[ofs * 3 + 0];
            float g = ((float *)ptr)[ofs * 3 + 1];
            float b = ((float *)ptr)[ofs * 3 + 2];
            return {r, g, b, 1.0};
        }
        case ImageFormat::RGBAF: {
            float r = ((float *)ptr)[ofs * 4 + 0];
            float g = ((float *)ptr)[ofs * 4 + 1];
            float b = ((float *)ptr)[ofs * 4 + 2];
            float a = ((float *)ptr)[ofs * 4 + 3];
            return {r, g, b, a};
        }
        case ImageFormat::RH: {
            uint16_t r = ((uint16_t *)ptr)[ofs];
            return {HalfToFloat(r), 0.0, 0.0, 1.0};
        }
        case ImageFormat::RGH: {
            uint16_t r = ((uint16_t *)ptr)[ofs * 2 + 0];
            uint16_t g = ((uint16_t *)ptr)[ofs * 2 + 1];
            return {HalfToFloat(r), HalfToFloat(g), 0.0, 1.0};
        }
        case ImageFormat::RGBH: {
            uint16_t r = ((uint16_t *)ptr)[ofs * 3 + 0];
            uint16_t g = ((uint16_t *)ptr)[ofs * 3 + 1];
            uint16_t b = ((uint16_t *)ptr)[ofs * 3 + 2];
            return {HalfToFloat(r), HalfToFloat(g), HalfToFloat(b), 1.0};
        }
        case ImageFormat::RGBAH: {
            uint16_t r = ((uint16_t *)ptr)[ofs * 4 + 0];
            uint16_t g = ((uint16_t *)ptr)[ofs * 4 + 1];
            uint16_t b = ((uint16_t *)ptr)[ofs * 4 + 2];
            uint16_t a = ((uint16_t *)ptr)[ofs * 4 + 3];
            return {HalfToFloat(r), HalfToFloat(g), HalfToFloat(b), HalfToFloat(a)};
        }
        case ImageFormat::RGBE9995: {
            return Color::FromRGBE9995(((uint32_t *)ptr)[ofs]);
        }
        default: {
            ERR_FAIL_V_MSG(Color(), "Can't get_pixel() on compressed image, sorry.");
        }
    }
}


void Image::SetColorAtOfs(uint8_t *ptr, uint32_t ofs, const Color &color) {
    switch (format_) {
        case ImageFormat::L8: {
            ptr[ofs] = uint8_t(std::clamp(color.val() * 255.0, 0.0, 255.0));
        } break;
        case ImageFormat::LA8: {
            ptr[ofs * 2 + 0] = uint8_t(std::clamp(color.val() * 255.0, 0.0, 255.0));
            ptr[ofs * 2 + 1] = uint8_t(std::clamp(color.alpha() * 255.0, 0.0, 255.0));
        } break;
        case ImageFormat::R8: {
            ptr[ofs] = uint8_t(std::clamp(color.red() * 255.0, 0.0, 255.0));
        } break;
        case ImageFormat::RG8: {
            ptr[ofs * 2 + 0] = uint8_t(std::clamp(color.red() * 255.0, 0.0, 255.0));
            ptr[ofs * 2 + 1] = uint8_t(std::clamp(color.green() * 255.0, 0.0, 255.0));
        } break;
        case ImageFormat::RGB8: {
            ptr[ofs * 3 + 0] = uint8_t(std::clamp(color.red() * 255.0, 0.0, 255.0));
            ptr[ofs * 3 + 1] = uint8_t(std::clamp(color.green() * 255.0, 0.0, 255.0));
            ptr[ofs * 3 + 2] = uint8_t(std::clamp(color.blue() * 255.0, 0.0, 255.0));
        } break;
        case ImageFormat::RGBA8: {
            ptr[ofs * 4 + 0] = uint8_t(std::clamp(color.red() * 255.0, 0.0, 255.0));
            ptr[ofs * 4 + 1] = uint8_t(std::clamp(color.green() * 255.0, 0.0, 255.0));
            ptr[ofs * 4 + 2] = uint8_t(std::clamp(color.blue() * 255.0, 0.0, 255.0));
            ptr[ofs * 4 + 3] = uint8_t(std::clamp(color.alpha() * 255.0, 0.0, 255.0));

        } break;
        case ImageFormat::RGBA4444: {
            uint16_t rgba = 0;

            rgba = uint16_t(std::clamp(color.red() * 15.0, 0.0, 15.0)) << 12;
            rgba |= uint16_t(std::clamp(color.green() * 15.0, 0.0, 15.0)) << 8;
            rgba |= uint16_t(std::clamp(color.blue() * 15.0, 0.0, 15.0)) << 4;
            rgba |= uint16_t(std::clamp(color.alpha() * 15.0, 0.0, 15.0));

            ((uint16_t *)ptr)[ofs] = rgba;

        } break;
        case ImageFormat::RGB565: {
            uint16_t rgba = 0;

            rgba = uint16_t(std::clamp(color.red() * 31.0, 0.0, 31.0));
            rgba |= uint16_t(std::clamp(color.green() * 63.0, 0.0, 33.0)) << 5;
            rgba |= uint16_t(std::clamp(color.blue() * 31.0, 0.0, 31.0)) << 11;

            ((uint16_t *)ptr)[ofs] = rgba;

        } break;
        case ImageFormat::RF: {
            ((float *)ptr)[ofs] = color.red();
        } break;
        case ImageFormat::RGF: {
            ((float *)ptr)[ofs * 2 + 0] = color.red();
            ((float *)ptr)[ofs * 2 + 1] = color.green();
        } break;
        case ImageFormat::RGBF: {
            ((float *)ptr)[ofs * 3 + 0] = color.red();
            ((float *)ptr)[ofs * 3 + 1] = color.green();
            ((float *)ptr)[ofs * 3 + 2] = color.blue();
        } break;
        case ImageFormat::RGBAF: {
            ((float *)ptr)[ofs * 4 + 0] = color.red();
            ((float *)ptr)[ofs * 4 + 1] = color.green();
            ((float *)ptr)[ofs * 4 + 2] = color.blue();
            ((float *)ptr)[ofs * 4 + 3] = color.alpha();
        } break;
        case ImageFormat::RH: {
            ((uint16_t *)ptr)[ofs] = FloatToHalf(color.red());
        } break;
        case ImageFormat::RGH: {
            ((uint16_t *)ptr)[ofs * 2 + 0] = FloatToHalf(color.red());
            ((uint16_t *)ptr)[ofs * 2 + 1] = FloatToHalf(color.green());
        } break;
        case ImageFormat::RGBH: {
            ((uint16_t *)ptr)[ofs * 3 + 0] = FloatToHalf(color.red());
            ((uint16_t *)ptr)[ofs * 3 + 1] = FloatToHalf(color.green());
            ((uint16_t *)ptr)[ofs * 3 + 2] = FloatToHalf(color.blue());
        } break;
        case ImageFormat::RGBAH: {
            ((uint16_t *)ptr)[ofs * 4 + 0] = FloatToHalf(color.red());
            ((uint16_t *)ptr)[ofs * 4 + 1] = FloatToHalf(color.green());
            ((uint16_t *)ptr)[ofs * 4 + 2] = FloatToHalf(color.blue());
            ((uint16_t *)ptr)[ofs * 4 + 3] = FloatToHalf(color.alpha());
        } break;
        case ImageFormat::RGBE9995: {
            ((uint32_t *)ptr)[ofs] = color.ToRGBE9995();

        } break;
        default: {
            ERR_FAIL_MSG("Can't set_pixel() on compressed image, sorry.");
        }
    }
}

Color Image::GetPixel(int x, int y) const {
    uint32_t ofs = y * width_ + x;
    return GetColorAtOfs(data_.data(), ofs);
}

void Image::SetPixel(int x, int y, const Color& color) {
    uint32_t ofs = y * width_ + x;
    return SetColorAtOfs(data_.data(), ofs, color);
}

Ref<Image> Image::LoadFromFile(const std::string &path)
{
    // TODO(wqq)
//    return gobot::dynamic_pointer_cast<Image>( ResourceFormatLoaderSDLImage::GetInstance()->Load(path));
    return {};
}

int Image::GetImageDataSize(int width, int height, ImageFormat format, bool mipmaps) {
    int mm;
    return GetDstImageSize(width, height, format, mm, mipmaps ? -1 : 0);
}

std::string Image::GetFormatName(ImageFormat format) {
    ERR_FAIL_INDEX_V((unsigned int)format, (unsigned int)ImageFormat::MAX, std::string());
    return s_format_names[(unsigned int)format];
}

std::vector<uint8_t> Image::GetData() const {
    return data_;
}

void Image::Fill(const Color &color) {
    // TODO(wqq)
}


int Image::GetFormatPixelSize(ImageFormat format) {
    switch (format) {
        case ImageFormat::L8:
            return 1; //luminance
        case ImageFormat::LA8:
            return 2; //luminance-alpha
        case ImageFormat::R8:
            return 1;
        case ImageFormat::RG8:
            return 2;
        case ImageFormat::RGB8:
            return 3;
        case ImageFormat::RGBA8:
            return 4;
        case ImageFormat::RGBA4444:
            return 2;
        case ImageFormat::RGB565:
            return 2;
        case ImageFormat::RF:
            return 4; //float
        case ImageFormat::RGF:
            return 8;
        case ImageFormat::RGBF:
            return 12;
        case ImageFormat::RGBAF:
            return 16;
        case ImageFormat::RH:
            return 2; //half float
        case ImageFormat::RGH:
            return 4;
        case ImageFormat::RGBH:
            return 6;
        case ImageFormat::RGBAH:
            return 8;
        case ImageFormat::RGBE9995:
            return 4;
        case ImageFormat::DXT1:
            return 1; //s3tc bc1
        case ImageFormat::DXT3:
            return 1; //bc2
        case ImageFormat::DXT5:
            return 1; //bc3
        case ImageFormat::RGTC_R:
            return 1; //bc4
        case ImageFormat::RGTC_RG:
            return 1; //bc5
        case ImageFormat::BPTC_RGBA:
            return 1; //btpc bc6h
        case ImageFormat::BPTC_RGBF:
            return 1; //float /
        case ImageFormat::BPTC_RGBFU:
            return 1; //unsigned float
        case ImageFormat::ETC:
            return 1; //etc1
        case ImageFormat::ETC2_R11:
            return 1; //etc2
        case ImageFormat::ETC2_R11S:
            return 1; //signed: return 1; NOT srgb.
        case ImageFormat::ETC2_RG11:
            return 1;
        case ImageFormat::ETC2_RG11S:
            return 1;
        case ImageFormat::ETC2_RGB8:
            return 1;
        case ImageFormat::ETC2_RGBA8:
            return 1;
        case ImageFormat::ETC2_RGB8A1:
            return 1;
        case ImageFormat::ETC2_RA_AS_RG:
            return 1;
        case ImageFormat::DXT5_RA_AS_RG:
            return 1;
        case ImageFormat::ASTC_4x4:
            return 1;
        case ImageFormat::ASTC_4x4_HDR:
            return 1;
        case ImageFormat::ASTC_8x8:
            return 1;
        case ImageFormat::ASTC_8x8_HDR:
            return 1;
        case ImageFormat::MAX: {
        }
    }
    return 0;
}

int Image::GetFormatBlockSize(ImageFormat format) {
    switch (format) {
        case ImageFormat::DXT1: //s3tc bc1
        case ImageFormat::DXT3: //bc2
        case ImageFormat::DXT5: //bc3
        case ImageFormat::RGTC_R: //bc4
        case ImageFormat::RGTC_RG: { //bc5		case case ImageFormat::DXT1:

            return 4;
        }
        case ImageFormat::ETC: {
            return 4;
        }
        case ImageFormat::BPTC_RGBA:
        case ImageFormat::BPTC_RGBF:
        case ImageFormat::BPTC_RGBFU: {
            return 4;
        }
        case ImageFormat::ETC2_R11: //etc2
        case ImageFormat::ETC2_R11S: //signed: NOT srgb.
        case ImageFormat::ETC2_RG11:
        case ImageFormat::ETC2_RG11S:
        case ImageFormat::ETC2_RGB8:
        case ImageFormat::ETC2_RGBA8:
        case ImageFormat::ETC2_RGB8A1:
        case ImageFormat::ETC2_RA_AS_RG: //used to make basis universal happy
        case ImageFormat::DXT5_RA_AS_RG: //used to make basis universal happy

        {
            return 4;
        }
        case ImageFormat::ASTC_4x4:
        case ImageFormat::ASTC_4x4_HDR: {
            return 4;
        }
        case ImageFormat::ASTC_8x8:
        case ImageFormat::ASTC_8x8_HDR: {
            return 8;
        }
        default: {
        }
    }

    return 1;
}

int Image::GetFormatPixelRshift(ImageFormat format) {
    if (format == ImageFormat::ASTC_8x8) {
        return 2;
    } else if (format == ImageFormat::DXT1 ||
               format == ImageFormat::RGTC_R ||
               format == ImageFormat::ETC ||
               format == ImageFormat::ETC2_R11 ||
               format == ImageFormat::ETC2_R11S ||
               format == ImageFormat::ETC2_RGB8 ||
               format == ImageFormat::ETC2_RGB8A1) {
        return 1;
    } else {
        return 0;
    }
}

void Image::GetFormatMinPixelSize(ImageFormat format, int &r_w, int &r_h) {
    switch (format) {
        case ImageFormat::DXT1: //s3tc bc1
        case ImageFormat::DXT3: //bc2
        case ImageFormat::DXT5: //bc3
        case ImageFormat::RGTC_R: //bc4
        case ImageFormat::RGTC_RG: { //bc5		case case ImageFormat::DXT1:
            r_w = 4;
            r_h = 4;
        } break;
        case ImageFormat::ETC: {
            r_w = 4;
            r_h = 4;
        } break;
        case ImageFormat::BPTC_RGBA:
        case ImageFormat::BPTC_RGBF:
        case ImageFormat::BPTC_RGBFU: {
            r_w = 4;
            r_h = 4;
        } break;
        case ImageFormat::ETC2_R11: //etc2
        case ImageFormat::ETC2_R11S: //signed: NOT srgb.
        case ImageFormat::ETC2_RG11:
        case ImageFormat::ETC2_RG11S:
        case ImageFormat::ETC2_RGB8:
        case ImageFormat::ETC2_RGBA8:
        case ImageFormat::ETC2_RGB8A1:
        case ImageFormat::ETC2_RA_AS_RG:
        case ImageFormat::DXT5_RA_AS_RG: {
            r_w = 4;
            r_h = 4;

        } break;
        case ImageFormat::ASTC_4x4:
        case ImageFormat::ASTC_4x4_HDR: {
            r_w = 4;
            r_h = 4;

        } break;
        case ImageFormat::ASTC_8x8:
        case ImageFormat::ASTC_8x8_HDR: {
            r_w = 8;
            r_h = 8;

        } break;
        default: {
            r_w = 1;
            r_h = 1;
        } break;
    }
}

int Image::GetDstImageSize(int width, int height, ImageFormat format, int& mipmaps,
                           int mipmaps_index, int* mm_width, int* mm_height)
{
    // Data offset in mipmaps (including the original texture).
    int size = 0;

    int w = width;
    int h = height;

    // Current mipmap index in the loop below. mipmaps_index is the target mipmap index.
    // In this function, mipmap 0 represents the first mipmap instead of the original texture.
    int mm = 0;

    int pixsize = GetFormatPixelSize(format);
    int pixshift = GetFormatPixelRshift(format);
    int block = GetFormatBlockSize(format);

    // Technically, you can still compress up to 1 px no matter the format, so commenting this.
    //int minw, minh;
    //get_format_min_pixel_size(format, minw, minh);
    int minw = 1, minh = 1;

    while (true) {
        int bw = w % block != 0 ? w + (block - w % block) : w;
        int bh = h % block != 0 ? h + (block - h % block) : h;

        int s = bw * bh;

        s *= pixsize;
        s >>= pixshift;

        size += s;

        if (mipmaps >= 0) {
            w = std::max(minw, w >> 1);
            h = std::max(minh, h >> 1);
        } else {
            if (w == minw && h == minh) {
                break;
            }
            w = std::max(minw, w >> 1);
            h = std::max(minh, h >> 1);
        }

        // Set mipmap size.
        if (mm_width) {
            *mm_width = w;
        }
        if (mm_height) {
            *mm_height = h;
        }

        // Reach target mipmap.
        if (mipmaps_index >= 0 && mm == mipmaps_index) {
            break;
        }

        mm++;
    }

    mipmaps = mm;
    return size;
}


void Image::InitializeData(int width, int height, bool use_mipmaps, ImageFormat format)
{
    ERR_FAIL_COND_MSG(width <= 0, "The Image width specified (" + std::to_string(width) + " pixels) must be greater than 0 pixels.");
    ERR_FAIL_COND_MSG(height <= 0, "The Image height specified (" + std::to_string(height) + " pixels) must be greater than 0 pixels.");
    ERR_FAIL_COND_MSG(width > MAX_WIDTH,
                      "The Image width specified (" + std::to_string(width) + " pixels) cannot be greater than " + std::to_string(MAX_WIDTH) + " pixels.");
    ERR_FAIL_COND_MSG(height > MAX_HEIGHT,
                      "The Image height specified (" + std::to_string(height) + " pixels) cannot be greater than " + std::to_string(MAX_HEIGHT) + " pixels.");
    ERR_FAIL_COND_MSG(width * height > MAX_PIXELS,
                      "Too many pixels for Image. Maximum is " + std::to_string(MAX_WIDTH) + "x" + std::to_string(MAX_HEIGHT) + " = " + std::to_string(MAX_PIXELS) + "pixels .");
    ERR_FAIL_COND_MSG(format >= ImageFormat::MAX, "The Image format specified (" + std::to_string(int(format)) + ") is out of range. See Image's Format enum.");

    int mm = 0;
    int size = GetDstImageSize(width, height, format, mm, use_mipmaps ? -1 : 0);
    data_.resize(size);

    {
        uint8_t *w = data_.data();
        memset(w, 0, size);
    }

    width_ = width;
    height_ = height;
    use_mipmaps_ = use_mipmaps;
    format_ = format;
}

void Image::InitializeData(int width, int height, bool use_mipmaps, ImageFormat format, const std::vector<uint8_t> &data) {
    ERR_FAIL_COND_MSG(width <= 0, "The Image width specified (" + std::to_string(width) + " pixels) must be greater than 0 pixels.");
    ERR_FAIL_COND_MSG(height <= 0, "The Image height specified (" + std::to_string(height) + " pixels) must be greater than 0 pixels.");
    ERR_FAIL_COND_MSG(width > MAX_WIDTH,
                      "The Image width specified (" + std::to_string(width) + " pixels) cannot be greater than " + std::to_string(MAX_WIDTH) + " pixels.");
    ERR_FAIL_COND_MSG(height > MAX_HEIGHT,
                      "The Image height specified (" + std::to_string(height) + " pixels) cannot be greater than " + std::to_string(MAX_HEIGHT) + " pixels.");
    ERR_FAIL_COND_MSG(width * height > MAX_PIXELS,
                      "Too many pixels for Image. Maximum is " + std::to_string(MAX_WIDTH) + "x" + std::to_string(MAX_HEIGHT) + " = " + std::to_string(MAX_PIXELS) + "pixels .");
    ERR_FAIL_COND_MSG(format >= ImageFormat::MAX, "The Image format specified (" + std::to_string(int(format)) + ") is out of range. See Image's Format enum.");

    int mm;
    int size = GetDstImageSize(width, height, format, mm, use_mipmaps ? -1 : 0);

    if (data.size() != size) [[unlikely]] {
        std::string description_mipmaps = GetFormatName(format) + " ";
        if (use_mipmaps) {
            const int num_mipmaps = GetImageRequiredMipmaps(width, height, format);
            if (num_mipmaps != 1) {
                description_mipmaps += fmt::format("with %d mipmaps", num_mipmaps);
            } else {
                description_mipmaps += "with 1 mipmap";
            }
        } else {
            description_mipmaps += "without mipmaps";
        }
        const std::string description = fmt::format("%dx%dx%d (%s)", width, height, GetFormatPixelSize(format), description_mipmaps);
        ERR_FAIL_MSG(fmt::format("Expected Image data size of %s = %d bytes, got %d bytes instead.", description, size, data.size()));
    }

    height_ = height;
    width_ = width;
    format_ = format;
    data_ = data;
    use_mipmaps_ = use_mipmaps;

}

std::array<Ref<Image>, 6> Image::ConvertEquirectangularMapToCubeMapFaces(const Ref<Image>& b) {
    const int face_size = b->GetWidth() / 4;

    const int w = face_size * 3;
    const int h = face_size * 4;

    std::array<Ref<Image>, 6> cube_map;
    for (int i = 0; i < cube_map.size(); i++ ) {
        cube_map[i] = MakeRef<Image>(face_size, face_size, false, b->GetFormat());
    }

    /*
        ------
        | +Y |
   ----------------
   | -X | -Z | +X |
   ----------------
        | -Y |
        ------
        | +Z |
        ------
    */
    std::array<Vector2i, 6> face_offsets = {
            {
                    {face_size, face_size * 3}, // +Z
                    {0, face_size},             // -X
                    {face_size, face_size},     // -Z
                    {face_size * 2, face_size}, // +X
                    {face_size, 0},             // +Y
                    {face_size, face_size * 2}  // -Y
            }
    };

    auto face_coords_to_XYZ = [](int i, int j, int faceID, int faceSize) -> Vector3f {
        const float A = 2.0f * float(i) / faceSize;
        const float B = 2.0f * float(j) / faceSize;

        if (faceID == 0) return {-1.0f, A - 1.0f, B - 1.0f};
        if (faceID == 1) return {A - 1.0f, -1.0f, 1.0f - B};
        if (faceID == 2) return {1.0f, A - 1.0f, 1.0f - B};
        if (faceID == 3) return {1.0f - A, 1.0f, 1.0f - B};
        if (faceID == 4) return {B - 1.0f, A - 1.0f, 1.0f};
        if (faceID == 5) return {1.0f - B, A - 1.0f, -1.0f};

        return {};
    };

    const int clamp_width = b->GetWidth() - 1;
    const int clamp_height = b->GetHeight() - 1;

    for (int face = 0; face != 6; face++) {
        for (int i = 0; i != face_size; i++) {
            for (int j = 0; j != face_size; j++) {
                const auto P = face_coords_to_XYZ(i, j, face, face_size);
                const float R = std::hypot(P.x(), P.y());
                const float theta = std::atan2(P.y(), P.x());
                const float phi = std::atan2(P.z(), R);
                //	float point source coordinates
                const auto Uf = float(2.0f * face_size * (theta + M_PI) / M_PI);
                const auto Vf = float(2.0f * face_size * (M_PI / 2.0f - phi) / M_PI);
                // 4-samples for bilinear interpolation
                const int U1 = std::clamp(int(floor(Uf)), 0, clamp_width);
                const int V1 = std::clamp(int(floor(Vf)), 0, clamp_height);
                const int U2 = std::clamp(U1 + 1, 0, clamp_width);
                const int V2 = std::clamp(V1 + 1, 0, clamp_height);
                // fractional part
                const float s = Uf - U1;
                const float t = Vf - V1;
                // fetch 4-samples
                const auto A = b->GetPixel(U1, V1);
                const auto B = b->GetPixel(U2, V1);
                const auto C = b->GetPixel(U1, V2);
                const auto D = b->GetPixel(U2, V2);
                // bilinear interpolation
                const auto color = A * (1 - s) * (1 - t) + B * (s) * (1 - t) + C * (1 - s) * t + D * (s) * (t);
                cube_map[face]->SetPixel(i + face_offsets[face].x(), j + face_offsets[face].y(), color);
            }
        };
    }

    return cube_map;
}


}