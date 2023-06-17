/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-13
*/


#include "gobot/drivers/opengl/gl_texture.hpp"
#include "gobot/scene/resources/bit_map.hpp"
#include "gobot/scene/resources/texture.hpp"

#include <stb_image.h>
#include <gli/gli.hpp>
#include <gli/texture2d.hpp>
#include <gli/load_ktx.hpp>

namespace gobot::opengl {

static int GetNumMipMapLevels2D(int w, int h)
{
    int levels = 1;
    while ((w | h) >> levels)
        levels += 1;
    return levels;
}

/// Draw a checkerboard on a pre-allocated square RGB image.
static uint8_t* GenerateDefaultCheckerboardImage(int* width, int* height)
{
    const int w = 128;
    const int h = 128;

    uint8_t* imgData = (uint8_t*)malloc(w * h * 3); // stbi_load() uses malloc(), so this is safe

    assert(imgData && w > 0 && h > 0);
    assert(w == h);

    if (!imgData || w <= 0 || h <= 0) return nullptr;
    if (w != h) return nullptr;

    for (int i = 0; i < w * h; i++)
    {
        const int row = i / w;
        const int col = i % w;
        imgData[i * 3 + 0] = imgData[i * 3 + 1] = imgData[i * 3 + 2] = 0xFF * ((row + col) % 2);
    }

    if (width) *width = w;
    if (height) *height = h;

    return imgData;
}

GLTexture::GLTexture(GLenum type, int width, int height, GLenum internal_format)
        : type_(type)
{
    glCreateTextures(type, 1, &handle_);
    glTextureParameteri(handle_, GL_TEXTURE_MAX_LEVEL, 0);
    glTextureParameteri(handle_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(handle_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureStorage2D(handle_, GetNumMipMapLevels2D(width, height), internal_format, width, height);
}

GLTexture::GLTexture(GLenum type, const char* file_name)
        : GLTexture(type, file_name, GL_REPEAT)
{}


GLTexture::GLTexture(GLenum type, const char* file_name, GLenum clamp)
        : type_(type)
{
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glCreateTextures(type, 1, &handle_);
    glTextureParameteri(handle_, GL_TEXTURE_MAX_LEVEL, 0);
    glTextureParameteri(handle_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(handle_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(handle_, GL_TEXTURE_WRAP_S, clamp);
    glTextureParameteri(handle_, GL_TEXTURE_WRAP_T, clamp);

    const char* ext = strrchr(file_name, '.');

    const bool isKTX = ext && !strcmp(ext, ".ktx");

    switch (type) {
        case GL_TEXTURE_2D: {
            int w = 0;
            int h = 0;
            int num_mipmaps = 0;
            if (isKTX) {
                gli::texture gliTex = gli::load_ktx(file_name);
                gli::gl GL(gli::gl::PROFILE_KTX);
                gli::gl::format const format = GL.translate(gliTex.format(), gliTex.swizzles());
                glm::tvec3<GLsizei> extent(gliTex.extent(0));
                w = extent.x;
                h = extent.y;
                num_mipmaps = GetNumMipMapLevels2D(w, h);
                glTextureStorage2D(handle_, num_mipmaps, format.Internal, w, h);
                glTextureSubImage2D(handle_, 0, 0, 0, w, h, format.External, format.Type, gliTex.data(0, 0, 0));
            } else {
                uint8_t* img = stbi_load(file_name, &w, &h, nullptr, STBI_rgb_alpha);

                // Note(Anton): replaced assert(img) with a fallback image to prevent crashes with missing files or bad (eg very long) paths.
                if (!img) {
                    fprintf(stderr, "WARNING: could not load image `%s`, using a fallback.\n", file_name);
                    img = GenerateDefaultCheckerboardImage(&w, &h);
                    if (!img)
                    {
                        fprintf(stderr, "FATAL ERROR: out of memory allocating image for fallback texture\n");
                        exit(EXIT_FAILURE);
                    }
                }

                num_mipmaps = GetNumMipMapLevels2D(w, h);
                glTextureStorage2D(handle_, num_mipmaps, GL_RGBA8, w, h);
                glTextureSubImage2D(handle_, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, img);
                stbi_image_free((void*)img);
            }
            glGenerateTextureMipmap(handle_);
            glTextureParameteri(handle_, GL_TEXTURE_MAX_LEVEL, num_mipmaps - 1);
            glTextureParameteri(handle_, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTextureParameteri(handle_, GL_TEXTURE_MAX_ANISOTROPY , 16);
            break;
        } case GL_TEXTURE_CUBE_MAP: {
            int w, h, comp;
            const float* img = stbi_loadf(file_name, &w, &h, &comp, 3);
            assert(img);
            BitMap in(w, h, comp, BitmapFormat::Float, img);
            const bool is_equirectangular = w == 2 * h;
            stbi_image_free((void*)img);
            auto cube_map = Image::ConvertEquirectangularMapToCubeMapFaces(nullptr);

//            const int num_mipmaps = GetNumMipMapLevels2D(cube_map.Width(), cube_map.Height());
//
//            glTextureParameteri(handle_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//            glTextureParameteri(handle_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//            glTextureParameteri(handle_, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
//            glTextureParameteri(handle_, GL_TEXTURE_BASE_LEVEL, 0);
//            glTextureParameteri(handle_, GL_TEXTURE_MAX_LEVEL, num_mipmaps - 1);
//            glTextureParameteri(handle_, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
//            glTextureParameteri(handle_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//            glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
//            glTextureStorage2D(handle_, num_mipmaps, GL_RGB32F, cube_map.Width(), cube_map.Height());
//            const uint8_t* data = cube_map.Data().data();
//
//            for (unsigned i = 0; i != 6; ++i) {
//                glTextureSubImage3D(handle_, 0, 0, 0, i, cube_map.Width(), cube_map.Height(), 1, GL_RGB, GL_FLOAT, data);
//                data += cube_map.Width() * cube_map.Height() * cube_map.Component() * BitMap::GetBytesPerComponent(cube_map.Format());
//            }

            glGenerateTextureMipmap(handle_);
            break;
        } default:
            assert(false);
    }

    handle_bindless_ = glGetTextureHandleARB(handle_);
    glMakeTextureHandleResidentARB(handle_bindless_);
}

GLTexture::GLTexture(int w, int h, const void* img)
        : type_(GL_TEXTURE_2D)
{
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glCreateTextures(type_, 1, &handle_);
    int num_mipmaps = GetNumMipMapLevels2D(w, h);
    glTextureStorage2D(handle_, num_mipmaps, GL_RGBA8, w, h);
    glTextureSubImage2D(handle_, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, img);
    glGenerateTextureMipmap(handle_);
    glTextureParameteri(handle_, GL_TEXTURE_MAX_LEVEL, num_mipmaps - 1);
    glTextureParameteri(handle_, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTextureParameteri(handle_, GL_TEXTURE_MAX_ANISOTROPY, 16);
    handle_bindless_ = glGetTextureHandleARB(handle_);
    glMakeTextureHandleResidentARB(handle_bindless_);
}

GLTexture::GLTexture(GLTexture&& other)
        : type_(other.type_)
        , handle_(other.handle_)
        , handle_bindless_(other.handle_bindless_)
{
    other.type_ = 0;
    other.handle_ = 0;
    other.handle_bindless_ = 0;
}

GLTexture::~GLTexture()
{
    if (handle_bindless_)
        glMakeTextureHandleNonResidentARB(handle_bindless_);
    glDeleteTextures(1, &handle_);
}


}
