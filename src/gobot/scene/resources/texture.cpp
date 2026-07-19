/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 23-3-17
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/texture.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/rendering/render_server.hpp"

namespace gobot {

Texture::~Texture() {
    if (RenderServer::HasInstance() && texture_rid_.IsValid()) {
        RS::GetInstance()->Free(texture_rid_);
    }
    texture_rid_ = RID();
}

RID Texture::GetRID() const {
    return texture_rid_;
}

Texture2D::Texture2D(const Ref<Image>& image) {
    SetImage(image);
}

void Texture2D::SetImage(const Ref<Image>& image) {
    if (image_.Get() == image.Get()) {
        return;
    }

    image_ = image;
    if (RenderServer::HasInstance() && image_.IsValid()) {
        if (texture_rid_.IsNull()) {
            texture_rid_ = RS::GetInstance()->TextureCreate();
            RS::GetInstance()->Texture2DInitialize(texture_rid_, image_);
        } else {
            RS::GetInstance()->TextureSetData(texture_rid_, image_);
        }
    }
    MarkChanged();
}

const Ref<Image>& Texture2D::GetImage() const {
    return image_;
}

void Texture2D::SetMinFilter(TextureFilter filter) {
    if (min_filter_ != filter) {
        min_filter_ = filter;
        MarkChanged();
    }
}

TextureFilter Texture2D::GetMinFilter() const {
    return min_filter_;
}

void Texture2D::SetMagFilter(TextureFilter filter) {
    if (mag_filter_ != filter) {
        mag_filter_ = filter;
        MarkChanged();
    }
}

TextureFilter Texture2D::GetMagFilter() const {
    return mag_filter_;
}

void Texture2D::SetWrapU(TextureWrap wrap) {
    if (wrap_u_ != wrap) {
        wrap_u_ = wrap;
        MarkChanged();
    }
}

TextureWrap Texture2D::GetWrapU() const {
    return wrap_u_;
}

void Texture2D::SetWrapV(TextureWrap wrap) {
    if (wrap_v_ != wrap) {
        wrap_v_ = wrap;
        MarkChanged();
    }
}

TextureWrap Texture2D::GetWrapV() const {
    return wrap_v_;
}

int Texture2D::GetWidth() const {
    return image_.IsValid() ? image_->GetWidth() : 0;
}

int Texture2D::GetHeight() const {
    return image_.IsValid() ? image_->GetHeight() : 0;
}

Vector2i Texture2D::GetSize() const {
    return image_.IsValid() ? image_->GetSize() : Vector2i::Zero();
}

Texture3D::Texture3D(uint16_t width, uint16_t height, std::uint16_t depth, bool has_mips,
                     TextureFormat format)
{
//    texture_rid_ = GET_RS()->CreateTexture3D(width, height, depth, has_mips, format, flags);
}

TextureCube::TextureCube(std::uint16_t size, bool has_mips, std::uint16_t num_layers,
                         TextureFormat format)

{
//    texture_rid_ = GET_RS()->CreateTextureCube(size, has_mips, num_layers, format, flags);
}



}

GOBOT_REGISTRATION {
    QuickEnumeration_<TextureFilter>("TextureFilter");
    QuickEnumeration_<TextureWrap>("TextureWrap");

    Class_<Texture>("Texture")
            .constructor()(CtorAsRawPtr);
    Type::register_wrapper_converter_for_base_classes<Ref<Texture>, Ref<Resource>>();

    Class_<Texture2D>("Texture2D")
            .constructor()(CtorAsRawPtr)
            .property("image", &Texture2D::GetImage, &Texture2D::SetImage)
            .property("min_filter", &Texture2D::GetMinFilter, &Texture2D::SetMinFilter)
            .property("mag_filter", &Texture2D::GetMagFilter, &Texture2D::SetMagFilter)
            .property("wrap_u", &Texture2D::GetWrapU, &Texture2D::SetWrapU)
            .property("wrap_v", &Texture2D::GetWrapV, &Texture2D::SetWrapV);
    Type::register_wrapper_converter_for_base_classes<Ref<Texture2D>, Ref<Texture>>();
    Type::register_wrapper_converter_for_base_classes<Ref<Texture2D>, Ref<Resource>>();
}
