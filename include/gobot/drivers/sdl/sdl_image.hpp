/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-13
*/

#pragma once

#include "gobot/core/types.hpp"

#include <gobot_export.h>
#include <SDL_image.h>

namespace gobot {

// Base from SDL_image
enum class SDLImageType {
    BMP,
    PNG,
    ICO,
    JPG,
    SVG,
    CUR,
    GIF,
    LBM,
    PCX,
    PNM,
    TIF,
    XCF,
    XPM,
    XV,
    WEBP,
    Unknown,
};

using SDLImage    = SDL_Surface;
using SDLStreamIO = SDL_RWops;

void FreeSDLImage(SDLImage* image);
using UniqueSDLImagePtr = std::unique_ptr<SDLImage, decltype(&FreeSDLImage)>;

class GOBOT_EXPORT SDLImageHandle {
public:
    static const std::vector<String> s_sdl_image_types;

    static SDLImageType GetSDLImageType(SDLStreamIO* sdl_stream_io);

    static UniqueSDLImagePtr LoadSDLImage(SDLImageType sdl_image_type, SDLStreamIO* sdl_stream_io);

};

}
