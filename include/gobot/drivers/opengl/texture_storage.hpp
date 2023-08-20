/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-13
*/

#pragma once

#include "glad/glad.h"
#include "gobot/core/io/image.hpp"
#include "gobot/core/rid.hpp"
#include "gobot/core/rid_owner.hpp"
#include "gobot/rendering/texture_storage.hpp"

namespace gobot::opengl {

class RenderTarget;

struct Texture {
    bool is_render_target = false;
    bool active = false;

    int total_data_size = 0;

    GLuint tex_id = 0; // opengl texture handle
    int width = 0;
    int height = 0;
    int depth = 0;
    int mipmaps = 1;
    int layers = 1;

    bool compressed = false;
    ImageFormat format = ImageFormat::R8;

    ImageFormat real_format = ImageFormat::R8;

    enum Type {
        Type_2D,
        Type_Layered,
        Type_3D
    };

    uint16_t stored_cube_sides = 0;

    Type type;

    GLenum target = GL_TEXTURE_2D;
    GLenum gl_format_cache = 0;
    GLenum gl_internal_format_cache = 0;
    GLenum gl_type_cache = 0;

    RenderTarget* render_target = nullptr;
};

struct RenderTarget {
    bool direct_to_screen = false;
    bool is_transparent = false;
    uint32_t view_count = 1;

    Vector2i size;

    GLuint fbo = 0; // frame buffer object

    RID texture;

    GLuint color = 0;
    GLuint color_internal_format = GL_RGBA8;
    GLuint color_format = GL_RGBA;
    GLuint color_type = GL_UNSIGNED_BYTE;
    Color clear_color{1.0f, 1.0f};
    ImageFormat image_format = ImageFormat::RGBA8;


    GLuint depth = 0;

};


class TextureStorage : public RendererTextureStorage {
public:
    TextureStorage();

    virtual ~TextureStorage();

    // render target related
    RID RenderTargetCreate() override;

    void RenderTargetFree(RID p_rid) override;

    inline bool OwnsRenderTarget(RID p_rid) { return render_target_owner_.Owns(p_rid); };

    void ClearRenderTarget(RenderTarget *rt);

    void RenderTargetSetSize(RID p_render_target, int p_width, int p_height, uint32_t p_view_count) override;

    RenderTarget* GetRenderTarget(RID p_rid) { return render_target_owner_.GetOrNull(p_rid); };

    void* GetRenderTargetColorTextureNativeHandle(RID p_render_target) override;

    // texture related

    RID TextureAllocate() override;

    void TextureFree(RID p_rid) override;

    bool OwnsTexture(RID p_rid) { return texture_owner_.Owns(p_rid); };

    inline Texture* GetTexture(RID rid) {
        Texture *texture = texture_owner_.GetOrNull(rid);
        return texture;
    }

    void Texture2DInitialize(RID texture_id, const Ref<Image> &image);

    void Texture2DPlaceholderInitialize(RID texture);

    void TextureSetData(RID texture_id, const Ref<Image> &image, int layer = 0);

    GLuint64 GetHandleBindless() const { return handle_bindless_; }

    String GetFramebufferError(GLenum p_status) {
#if defined(DEBUG_ENABLED)
        if (p_status == GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT) {
		return "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
	} else if (p_status == GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT) {
		return "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
	} else if (p_status == GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER) {
		return "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER";
	} else if (p_status == GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER) {
		return "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER";
	}
#endif
        return {std::to_string(p_status).c_str()};
    }


public:
    const static GLuint s_system_fbo;

    static TextureStorage* GetInstance();

private:
    static TextureStorage *s_singleton;

    Ref<Image> GetGLImageAndFormat(const Ref<Image> &p_image, ImageFormat p_format, ImageFormat &r_real_format,
                                   GLenum &r_gl_format, GLenum &r_gl_internal_format, GLenum &r_gl_type,
                                   bool &r_compressed, bool p_force_decompress) const;

    void UpdateRenderTarget(RenderTarget* render_target);

    void TextureSetData(RID p_texture, const Ref<Image> &p_image, int p_layer, bool initialize);

private:
    GLuint64 handle_bindless_ = 0;

    mutable RID_Owner<Texture> texture_owner_;

    mutable RID_Owner<RenderTarget> render_target_owner_;

};


}
