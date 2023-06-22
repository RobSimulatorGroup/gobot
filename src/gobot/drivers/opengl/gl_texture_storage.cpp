/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-13
*/

#include "gobot/drivers/opengl/gl_texture_storage.hpp"
#include "gobot/scene/resources/texture.hpp"

#include <stb_image.h>
#include <gli/gli.hpp>
#include <gli/texture2d.hpp>

namespace gobot::opengl {

TextureStorage *TextureStorage::s_singleton = nullptr;

const GLuint TextureStorage::s_system_fbo = 0;

static const GLenum s_cube_side_enum[6] = {
        GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
        GL_TEXTURE_CUBE_MAP_POSITIVE_X,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
};

TextureStorage* TextureStorage::GetInstance() {
    return s_singleton;
}

TextureStorage::TextureStorage() {
    s_singleton = this;
}

TextureStorage::~TextureStorage() {
    s_singleton = nullptr;

}

RID TextureStorage::TextureAllocate() {
    return texture_owner_.AllocateRID();
}

RID TextureStorage::RenderTargetCreate() {
    RenderTarget render_target;

    Texture t;
    t.active = true;
    t.render_target = &render_target;
    t.is_render_target = true;

    render_target.texture = texture_owner_.MakeRID(t);
    UpdateRenderTarget(&render_target);
    return render_target_owner_.MakeRID(render_target);
}

void TextureStorage::RenderTargetFree(RID p_rid) {
    RenderTarget *rt = render_target_owner_.GetOrNull(p_rid);
    ClearRenderTarget(rt);

    Texture *t = GetTexture(rt->texture);
    if (t) {
        t->is_render_target = false;
    }
    render_target_owner_.Free(p_rid);
}

void TextureStorage::ClearRenderTarget(RenderTarget *rt) {
    // there is nothing to clear when DIRECT_TO_SCREEN is used
    if (rt->direct_to_screen) {
        return;
    }

    if (rt->fbo) {
        glDeleteFramebuffers(1, &rt->fbo);
        rt->fbo = 0;
    }

    if (rt->texture.IsValid()) {
        Texture *tex = GetTexture(rt->texture);
        tex->width = 0;
        tex->height = 0;
        tex->active = false;
        tex->render_target = nullptr;
        tex->is_render_target = false;
        tex->tex_id = 0;
    }

    glDeleteTextures(1, &rt->color);
    rt->color = 0;

    glDeleteTextures(1, &rt->depth);
    rt->depth = 0;
}

void TextureStorage::Texture2DPlaceholderInitialize(RID texture) {
    //this could be better optimized to reuse an existing image , done this way
    //for now to get it working
    Ref<Image> image = MakeRef<Image>(4, 4, false, ImageFormat::RGBA8);
    image->Fill(Color(1, 0, 1, 1));
    Texture2DInitialize(texture, image);
}

void TextureStorage::TextureSetData(RID texture_id, const Ref<Image> &image, int layer) {
    TextureSetData(texture_id, image, layer, false);
}


void TextureStorage::TextureSetData(RID p_texture, const Ref<Image> &p_image, int p_layer, bool initialize) {
    Texture *texture = texture_owner_.GetOrNull(p_texture);

    ERR_FAIL_COND(!texture);
    if (texture->target == GL_TEXTURE_3D) {
        // Target is set to a 3D texture or array texture, exit early to avoid spamming errors
        return;
    }
    ERR_FAIL_COND(!texture->active);
    ERR_FAIL_COND(texture->is_render_target);
    ERR_FAIL_COND(!p_image.IsValid());
    ERR_FAIL_COND(texture->format != p_image->GetFormat());

    ERR_FAIL_COND(!p_image->GetWidth());
    ERR_FAIL_COND(!p_image->GetHeight());

    GLenum type;
    GLenum format;
    GLenum internal_format;
    bool compressed = false;

    ImageFormat real_format;
    Ref<Image> img = GetGLImageAndFormat(p_image, p_image->GetFormat(), real_format, format, internal_format, type, compressed,
                                         false);
    ERR_FAIL_COND(!img);

    std::vector<uint8_t> read = img->GetData();

    glTextureParameteri(texture->tex_id, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(texture->tex_id, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(texture->tex_id, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTextureParameteri(texture->tex_id, GL_TEXTURE_WRAP_T, GL_REPEAT);

    int mipmaps = img->HasMipmaps() ? img->GetMipmapCount() + 1 : 1;

    int w = img->GetWidth();
    int h = img->GetHeight();

    int tsize = 0;

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if (texture->target == GL_TEXTURE_2D_ARRAY) {
        if (initialize) {
            glTextureStorage3D(texture->tex_id, mipmaps, internal_format, w, h, texture->layers);
        }
    } else  {
        glTextureStorage2D(texture->tex_id, mipmaps, internal_format, w, h);
    }
    for (int i = 0; i < mipmaps; i++) {
        int size, ofs;
        img->GetMipmapOffsetAndSize(i, ofs, size);

        if (texture->target == GL_TEXTURE_2D_ARRAY) {
            glTextureSubImage3D(texture->tex_id, i, 0, 0, 0, w, h, p_layer, format, type, &read[ofs]);
        } else {
            glTextureSubImage2D(texture->tex_id, i, 0, 0, w, h, internal_format, type, &read[ofs]);
        }

        tsize += size;

        w = std::max(1, w >> 1);
        h = std::max(1, h >> 1);
    }

    texture->total_data_size = tsize;

    texture->mipmaps = mipmaps;
}




void TextureStorage::Texture2DInitialize(RID texture_id, const Ref<Image> &image) {
    ERR_FAIL_COND(!image.IsValid());

    Texture texture;
    texture.width = image->GetWidth();
    texture.height = image->GetHeight();
    texture.mipmaps = image->GetMipmapCount() + 1;
    texture.format = image->GetFormat();
    texture.type = Texture::Type_2D;
    texture.target = GL_TEXTURE_2D;
    GetGLImageAndFormat(Ref<Image>(), texture.format, texture.real_format, texture.gl_format_cache,
                        texture.gl_internal_format_cache, texture.gl_type_cache, texture.compressed, false);
    texture.total_data_size = image->GetImageDataSize(texture.width, texture.height, texture.format, texture.mipmaps);
    texture.active = true;
    glGenTextures(1, &texture.tex_id);
    texture_owner_.InitializeRID(texture_id, texture);
    TextureSetData(texture_id, image);
}


Ref<Image> TextureStorage::GetGLImageAndFormat(const Ref<Image> &p_image, ImageFormat p_format, ImageFormat &r_real_format,
                                               GLenum &r_gl_format, GLenum &r_gl_internal_format, GLenum &r_gl_type,
                                               bool &r_compressed, bool p_force_decompress) const {
    switch (p_format) {
        case ImageFormat::L8: {
            r_gl_internal_format = GL_R8;
			r_gl_format = GL_RED;
			r_gl_type = GL_UNSIGNED_BYTE;
        } break;
        case ImageFormat::LA8: {
            r_gl_internal_format = GL_RG8;
			r_gl_format = GL_RG;
			r_gl_type = GL_UNSIGNED_BYTE;
        } break;
        case ImageFormat::R8: {
            r_gl_internal_format = GL_R8;
            r_gl_format = GL_RED;
            r_gl_type = GL_UNSIGNED_BYTE;

        } break;
        case ImageFormat::RG8: {
            r_gl_internal_format = GL_RG8;
            r_gl_format = GL_RG;
            r_gl_type = GL_UNSIGNED_BYTE;

        } break;
        case ImageFormat::RGB8: {
            r_gl_internal_format = GL_RGB8;
            r_gl_format = GL_RGB;
            r_gl_type = GL_UNSIGNED_BYTE;

        } break;
        case ImageFormat::RGBA8: {
            r_gl_format = GL_RGBA;
            r_gl_internal_format = GL_RGBA8;
            r_gl_type = GL_UNSIGNED_BYTE;

        } break;
        case ImageFormat::RGBA4444: {
            r_gl_internal_format = GL_RGBA4;
            r_gl_format = GL_RGBA;
            r_gl_type = GL_UNSIGNED_SHORT_4_4_4_4;

        } break;
        case ImageFormat::RF: {
            r_gl_internal_format = GL_R32F;
            r_gl_format = GL_RED;
            r_gl_type = GL_FLOAT;

        } break;
        case ImageFormat::RGF: {
            r_gl_internal_format = GL_RG32F;
            r_gl_format = GL_RG;
            r_gl_type = GL_FLOAT;

        } break;
        case ImageFormat::RGBF: {
            r_gl_internal_format = GL_RGB32F;
            r_gl_format = GL_RGB;
            r_gl_type = GL_FLOAT;

        } break;
        case ImageFormat::RGBAF: {
            r_gl_internal_format = GL_RGBA32F;
            r_gl_format = GL_RGBA;
            r_gl_type = GL_FLOAT;

        } break;
        case ImageFormat::RH: {
            r_gl_internal_format = GL_R16F;
            r_gl_format = GL_RED;
            r_gl_type = GL_HALF_FLOAT;
        } break;
        case ImageFormat::RGH: {
            r_gl_internal_format = GL_RG16F;
            r_gl_format = GL_RG;
            r_gl_type = GL_HALF_FLOAT;

        } break;
        case ImageFormat::RGBH: {
            r_gl_internal_format = GL_RGB16F;
            r_gl_format = GL_RGB;
            r_gl_type = GL_HALF_FLOAT;

        } break;
        case ImageFormat::RGBAH: {
            r_gl_internal_format = GL_RGBA16F;
            r_gl_format = GL_RGBA;
            r_gl_type = GL_HALF_FLOAT;

        } break;
        case ImageFormat::RGBE9995: {
            r_gl_internal_format = GL_RGB9_E5;
            r_gl_format = GL_RGB;
            r_gl_type = GL_UNSIGNED_INT_5_9_9_9_REV;

        } break;
        default: {
            ERR_FAIL_V_MSG(Ref<Image>(), fmt::format("Image Format: is not supported by the OpenGL46 Renderer",
                                                     magic_enum::enum_name(p_format).data()).data());
        }
    }


    return p_image;
}

// use OpenGL Direct State Access (DSA)
void TextureStorage::UpdateRenderTarget(RenderTarget* rt) {
    // do not allocate a render target with no size
    if (rt->size.x() <= 0 || rt->size.y() <= 0) {
        return;
    }

    // do not allocate a render target that is attached to the screen
    if (rt->direct_to_screen) {
        rt->fbo = s_system_fbo;
        return;
    }

    rt->color_internal_format = rt->is_transparent ? GL_RGBA8 : GL_RGB10_A2;
    rt->color_format = GL_RGBA;
    rt->color_type = rt->is_transparent ? GL_UNSIGNED_BYTE : GL_UNSIGNED_INT_2_10_10_10_REV;
    rt->image_format = ImageFormat::RGBA8;

    // TODO(wqq): Do we need this.
    glDisable(GL_SCISSOR_TEST);
    glColorMask(1, 1, 1, 1);
    glDepthMask(GL_FALSE);

    {
        // TODO(wqq) support use_multiview
        GLenum texture_target = GL_TEXTURE_2D;

        // create fbo
        glCreateFramebuffers(1, &rt->fbo);

        // init attachment color
        auto* texture = GetTexture(rt->texture);
        ERR_FAIL_COND(!texture);

        glCreateTextures(texture_target, 1, &rt->color);
        glTextureParameteri(rt->color, GL_TEXTURE_MAX_LEVEL, 0);
        glTextureParameteri(rt->color, GL_TEXTURE_BASE_LEVEL, 0);
        glTextureParameteri(rt->color, GL_TEXTURE_MAX_LEVEL, 0);
        glTextureParameteri(rt->color, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(rt->color, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureParameteri(rt->color, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(rt->color, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glTextureStorage2D(rt->color, 1, rt->color_internal_format, rt->size.x(), rt->size.y());
        glNamedFramebufferTexture(rt->fbo, GL_COLOR_ATTACHMENT0, rt->color, 0);

        // init attachment depth
        glCreateTextures(texture_target, 1, &rt->depth);

        glTextureParameteri(rt->depth, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureParameteri(rt->depth, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(rt->depth, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(rt->depth, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glTextureStorage2D(rt->depth, 1, GL_DEPTH_COMPONENT24, rt->size.x(), rt->size.y());
        glNamedFramebufferTexture(rt->fbo, GL_DEPTH_ATTACHMENT, rt->depth, 0);

        GLenum status = glCheckNamedFramebufferStatus(rt->fbo, GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            // bind to default frame buffer
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glDeleteFramebuffers(1, &rt->fbo);
            glDeleteTextures(1, &rt->color);
            rt->fbo = 0;
            rt->size.x() = 0;
            rt->size.y() = 0;
            rt->color = 0;
            rt->depth = 0;
            LOG_WARN("Could not create render target, status: " + GetFramebufferError(status));
            return;
        }

        texture->is_render_target = true;
        texture->render_target = rt;
        texture->tex_id = rt->color;
        texture->active = true;
    }

    // TODO(wqq): Do we need this.
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, s_system_fbo);
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



}
