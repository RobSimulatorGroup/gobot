/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-10
*/

#pragma once

#include "gobot/core/io/resource.hpp"
#include "gobot/drivers/sdl/sdl_image.hpp"
#include <Eigen/Dense>

namespace gobot {

class Image : public Resource {
    GOBCLASS(Image, Resource);
public:
    Image();

    int GetWidth() const;

    int GetHeight() const;

    Eigen::Vector2i GetSize() const;

    FORCE_INLINE void SetSDLImage(UniqueSDLImagePtr sdl_image) { sdl_image_ = std::move(sdl_image); }

    FORCE_INLINE bool IsSDLImage() const { return sdl_image_ != nullptr; }

    FORCE_INLINE const SDLImage* GetSDLImage() const {  return sdl_image_.get(); }

    FORCE_INLINE SDLImage* GetSDLImage() {  return sdl_image_.get(); }

    static Ref<Image> LoadFromFile(const String &path);

private:
    UniqueSDLImagePtr sdl_image_{nullptr, &FreeSDLImage};

};

}