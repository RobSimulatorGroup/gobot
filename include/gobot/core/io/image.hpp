/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-10
*/

#pragma once

#include "gobot/core/io/resource.hpp"
#include "gobot/core/math/matrix.hpp"
#include "gobot/core/color.hpp"

namespace gobot {

enum class ImageFormat {
    L8, //luminance
    LA8, //luminance-alpha
    R8,
    RG8,
    RGB8,
    RGBA8,
    RGBA4444,
    RGB565,
    RF, //float
    RGF,
    RGBF,
    RGBAF,
    RH, //half float
    RGH,
    RGBH,
    RGBAH,
    RGBE9995,
    DXT1, //s3tc bc1
    DXT3, //bc2
    DXT5, //bc3
    RGTC_R,
    RGTC_RG,
    BPTC_RGBA, //btpc bc7
    BPTC_RGBF, //float bc6h
    BPTC_RGBFU, //unsigned float bc6hu
    ETC, //etc1
    ETC2_R11, //etc2
    ETC2_R11S, //signed, NOT srgb.
    ETC2_RG11,
    ETC2_RG11S,
    ETC2_RGB8,
    ETC2_RGBA8,
    ETC2_RGB8A1,
    ETC2_RA_AS_RG, //used to make basis universal happy
    DXT5_RA_AS_RG, //used to make basis universal happy
    ASTC_4x4,
    ASTC_4x4_HDR,
    ASTC_8x8,
    ASTC_8x8_HDR,
    MAX
};

class Image;

using ImageMemLoadFunc = Ref<Image>(*)(const uint8_t *image, int size);

class GOBOT_EXPORT Image : public Resource {
    GOBCLASS(Image, Resource);
public:
    enum {
        MAX_WIDTH = (1 << 24), // force a limit somehow
        MAX_HEIGHT = (1 << 24), // force a limit somehow
        MAX_PIXELS = 268435456
    };

    enum Interpolation {
        INTERPOLATE_NEAREST,
        INTERPOLATE_BILINEAR,
        INTERPOLATE_CUBIC,
        INTERPOLATE_TRILINEAR,
        INTERPOLATE_LANCZOS,
    };

    // create empty image
    Image();

    // create an empty image of a specific size and format
    Image(int width, int height, bool use_mipmaps, ImageFormat format);

    // import an image of a specific size and format from a pointer
    Image(int width, int height, bool use_mipmaps, ImageFormat format, const std::vector<uint8_t>& data);

    int GetWidth() const;

    int GetHeight() const;

    Vector2i GetSize() const;

    bool HasMipmaps() const;

    int GetMipmapCount() const;

    ImageFormat GetFormat() const;

    // get where the mipmap begins in data
    int GetMipmapByteSize(int mipmap) const;

    // get where the mipmap begins in data
    int GetMipmapOffset(int mipmap_index) const;

    // get where the mipmap begins in data
    void GetMipmapOffsetAndSize(int mipmap_index, int &r_ofs, int &r_size) const;

    // get where the mipmap begins in data
    void GetMipmapOffsetSizeAndDimensions(int mipmap_index, int &r_ofs, int &r_size, int &w, int &h) const;

    void InitializeData(int width, int height, bool use_mipmaps, ImageFormat format);

    void InitializeData(int width, int height, bool use_mipmaps, ImageFormat format, const std::vector<uint8_t> &data);


public:
    std::vector<uint8_t> GetData() const;

    static Ref<Image> LoadFromFile(const String &path);

    static ImageMemLoadFunc s_png_mem_loader_func;

    static ImageMemLoadFunc s_jpg_mem_loader_func;

public:
    static String GetFormatName(ImageFormat format);

    static int GetFormatPixelSize(ImageFormat format);

    static int GetFormatBlockSize(ImageFormat format);

    static int GetFormatPixelRshift(ImageFormat format);

    static void GetFormatMinPixelSize(ImageFormat format, int &r_w, int &r_h);

    static int GetImageRequiredMipmaps(int width, int height, ImageFormat format);

    static const char* s_format_names[(unsigned int)ImageFormat::MAX];

public:
    void Rotate90();

    void Rotate180();

    void FlipX();

    void FlipY();

    Color GetPixel(int x, int y) const;

    void SetPixel(int x, int y,  const Color& color);

public:
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
    // Map follow the sequence of [+Z, -X, -Z, +X, +Y, -Y]
    static std::array<Ref<Image>, 6> ConvertEquirectangularMapToCubeMapFaces(const Ref<Image>& b);

private:
    // if mipmaps_index is -1, the mipmaps
    // mipmaps_index is the target mipmap index, mipmaps_index 0 represents the first mipmap instead of the original texture
    static int GetDstImageSize(int width, int height, ImageFormat format, int& mipmaps,
                               int mipmaps_index = -1, int* mipmaps_width = nullptr, int* mipmaps_height = nullptr);

    // get where the mipmap begins in data
    FORCE_INLINE void GetMipmapOffsetAndSize(int mipmap, int &r_offset, int &r_width, int &r_height) const;

    FORCE_INLINE Color GetColorAtOfs(const uint8_t *ptr, uint32_t ofs) const;

    FORCE_INLINE void SetColorAtOfs(uint8_t *ptr, uint32_t ofs, const Color &color);

private:
    ImageFormat format_{ImageFormat::L8};
    std::vector<uint8_t> data_;
    int width_{0};
    int height_{0};
    bool use_mipmaps_{false};
};

}