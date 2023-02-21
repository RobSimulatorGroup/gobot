/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-10
*/


#include "gobot/core/io/image.hpp"
#include "gobot/core/io/image_load.hpp"

namespace gobot {

Image::Image()
{

}

int Image::GetWidth() const {
    if (sdl_image_)
        return sdl_image_->w;
    return 0;
}

int Image::GetHeight() const {
    if (sdl_image_)
        return sdl_image_->h;
    return 0;
}

Vector2i Image::GetSize() const {
    if (sdl_image_)
        return {sdl_image_->w , sdl_image_->h};
    return {0, 0};
}

Ref<Image> Image::LoadFromFile(const String &path)
{
    return gobot::dynamic_pointer_cast<Image>( ResourceFormatLoaderSDLImage::GetInstance()->Load(path));
}

}