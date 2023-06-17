/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-13
*/

#pragma once

#include "gobot/core/io/resource.hpp"
#include "gobot/core/math/matrix.hpp"

namespace gobot {

enum class BitmapType {
    Type2D,
    TypeCube
};

enum class BitmapFormat {
    UnsignedByte,
    Float,
};

class BitMap : public Resource {
    GOBCLASS(BitMap, Resource)
public:
    BitMap() = default;

    BitMap(int width, int height, int comp, BitmapFormat fmt)
            : width_(width),
              height_(height),
              component_(comp),
              fmt_(fmt),
              data_(width_ * height_ * comp * GetBytesPerComponent(fmt))
    {
        InitGetSetFuncs();
    }

    BitMap(int width, int height, int depth, int comp, BitmapFormat fmt)
            : width_(width),
              height_(height),
              depth_(depth),
              component_(comp),
              fmt_(fmt),
              data_(width_ * height_ * depth_ * comp * GetBytesPerComponent(fmt))
    {
        InitGetSetFuncs();
    }

    BitMap(int width, int height, int comp, BitmapFormat fmt, const void* ptr)
            : width_(width),
              height_(height),
              component_(comp),
              fmt_(fmt),
              data_(width_ * height_ * comp * GetBytesPerComponent(fmt))
    {
        InitGetSetFuncs();
        memcpy(data_.data(), ptr, data_.size());
    }

    static int GetBytesPerComponent(BitmapFormat fmt)
    {
        if (fmt == BitmapFormat::UnsignedByte) return 1;
        if (fmt == BitmapFormat::Float) return 4;
        return 0;
    }

    inline void SetPixel(int x, int y, const Vector4f& c)
    {
        std::invoke(set_pixel_func_, this, x, y, c);
    }

    inline Vector4f GetPixel(int x, int y) const
    {
        return std::invoke(get_pixel_func_, this, x, y);
    }

    inline int& Width() { return width_; }

    inline const int& Width() const { return width_; }

    inline int& Height() { return height_; }

    inline const int& Height() const { return height_; }

    inline int& Depth() { return depth_; }

    inline const int& Depth() const { return depth_; }

    inline int& Component() { return component_; }

    inline const int& Component() const { return component_; }

    inline BitmapType& DataType() { return type_; }

    inline const BitmapType& DataType() const { return type_; }

    inline BitmapFormat& Format() { return fmt_; }

    inline const BitmapFormat& Format() const { return fmt_; }

    std::vector<uint8_t>& Data() { return data_; }

    const std::vector<uint8_t>& Data() const { return data_; }

private:
    using SetPixelFunc = void(BitMap::*)(int, int, const Vector4f&);
    using GetPixelFunc = Vector4f(BitMap::*)(int, int) const;

    void InitGetSetFuncs() {
        switch (fmt_) {
            case BitmapFormat::UnsignedByte:
                set_pixel_func_ = &BitMap::SetPixelUnsignedByte;
                get_pixel_func_ = &BitMap::GetPixelUnsignedByte;
                break;
            case BitmapFormat::Float:
                set_pixel_func_ = &BitMap::SetPixelFloat;
                get_pixel_func_ = &BitMap::GetPixelFloat;
                break;
        }
    }

    void SetPixelFloat(int x, int y, const Vector4f& c)
    {
        const int ofs = component_ * (y * width_ + x);
        auto* data = reinterpret_cast<float*>(data_.data());
        if (component_ > 0) data[ofs + 0] = c.x();
        if (component_ > 1) data[ofs + 1] = c.y();
        if (component_ > 2) data[ofs + 2] = c.z();
        if (component_ > 3) data[ofs + 3] = c.w();
    }

    Vector4f GetPixelFloat(int x, int y) const
    {
        const int ofs = component_ * (y * width_ + x);
        const auto* data = reinterpret_cast<const float*>(data_.data());
        return {component_ > 0 ? data[ofs + 0] : 0.0f,
                component_ > 1 ? data[ofs + 1] : 0.0f,
                component_ > 2 ? data[ofs + 2] : 0.0f,
                component_ > 3 ? data[ofs + 3] : 0.0f};
    }

    inline void SetPixelUnsignedByte(int x, int y, const Vector4f& c)
    {
        const int ofs = component_ * (y * width_ + x);
        if (component_ > 0) data_[ofs + 0] = uint8_t(c.x() * 255.0f);
        if (component_ > 1) data_[ofs + 1] = uint8_t(c.y() * 255.0f);
        if (component_ > 2) data_[ofs + 2] = uint8_t(c.z() * 255.0f);
        if (component_ > 3) data_[ofs + 3] = uint8_t(c.w() * 255.0f);
    }

    Vector4f GetPixelUnsignedByte(int x, int y) const
    {
        const int ofs = component_ * (y * width_ + x);
        return {component_ > 0 ? float(data_[ofs + 0]) / 255.0f : 0.0f,
                component_ > 1 ? float(data_[ofs + 1]) / 255.0f : 0.0f,
                component_ > 2 ? float(data_[ofs + 2]) / 255.0f : 0.0f,
                component_ > 3 ? float(data_[ofs + 3]) / 255.0f : 0.0f};
    }

private:
    int width_ = 0;
    int height_ = 0;
    int depth_ = 1;

    // GL_RED/GL_RGB/GL_RGBA
    int component_ = 3;
    BitmapFormat fmt_{BitmapFormat::UnsignedByte};
    BitmapType type_{BitmapType::Type2D};
    std::vector<uint8_t> data_;

    SetPixelFunc set_pixel_func_ = &BitMap::SetPixelUnsignedByte;
    GetPixelFunc get_pixel_func_ = &BitMap::GetPixelUnsignedByte;
};



}
