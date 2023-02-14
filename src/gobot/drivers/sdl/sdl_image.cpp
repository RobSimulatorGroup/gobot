/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-13
*/


#include "gobot/drivers/sdl/sdl_image.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

const std::vector<String> SDLImageHandle::s_sdl_image_types = {
        "bmp",
        "png",
        "ico",
        "jpg",
        "svg",
        "cur",
        "gif",
        "lbm",
        "pcx",
        "pnm",
        "tif",
        "xcf",
        "xpm",
        "xv",
        "webp",
};

void FreeSDLImage(SDLImage* image)
{
    if (image)
        SDL_FreeSurface(image);
}

SDLImageType SDLImageHandle::GetSDLImageType(SDLStreamIO* sdl_stream_io)
{
    // Returns non-zero if this is XXX data, zero otherwise.
    int res = IMG_isBMP(sdl_stream_io);
    if (res) return SDLImageType::BMP;

    res = IMG_isPNG(sdl_stream_io);
    if (res) return SDLImageType::PNG;

    res = IMG_isICO(sdl_stream_io);
    if (res) return SDLImageType::ICO;

    res = IMG_isJPG(sdl_stream_io);
    if (res) return SDLImageType::JPG;

    res = IMG_isSVG(sdl_stream_io);
    if (res) return SDLImageType::SVG;

    res = IMG_isCUR(sdl_stream_io);
    if (res) return SDLImageType::CUR;

    res = IMG_isGIF(sdl_stream_io);
    if (res) return SDLImageType::GIF;

    res = IMG_isLBM(sdl_stream_io);
    if (res) return SDLImageType::LBM;

    res = IMG_isPCX(sdl_stream_io);
    if (res) return SDLImageType::PCX;

    res = IMG_isPNM(sdl_stream_io);
    if (res) return SDLImageType::PNM;

    res = IMG_isTIF(sdl_stream_io);
    if (res) return SDLImageType::TIF;

    res = IMG_isXCF(sdl_stream_io);
    if (res) return SDLImageType::XCF;

    res = IMG_isXPM(sdl_stream_io);
    if (res) return SDLImageType::XPM;

    res = IMG_isXV(sdl_stream_io);
    if (res) return SDLImageType::XV;

    res = IMG_isWEBP(sdl_stream_io);
    if (res) return SDLImageType::WEBP;

    return SDLImageType::Unknown;
}

UniqueSDLImage SDLImageHandle::LoadSDLImage(SDLImageType sdl_image_type, SDLStreamIO* sdl_stream_io)
{
    switch (sdl_image_type) {
        case SDLImageType::BMP:
            return {IMG_LoadBMP_RW(sdl_stream_io), &FreeSDLImage};
        case SDLImageType::PNG:
            return {IMG_LoadPNG_RW(sdl_stream_io), &FreeSDLImage};
        case SDLImageType::ICO:
            return {IMG_LoadICO_RW(sdl_stream_io), &FreeSDLImage};
        case SDLImageType::JPG:
            return {IMG_LoadJPG_RW(sdl_stream_io), &FreeSDLImage};
        case SDLImageType::SVG:
            return {IMG_LoadSVG_RW(sdl_stream_io), &FreeSDLImage};
        case SDLImageType::CUR:
            return {IMG_LoadCUR_RW(sdl_stream_io), &FreeSDLImage};
        case SDLImageType::GIF:
            return {IMG_LoadGIF_RW(sdl_stream_io), &FreeSDLImage};
        case SDLImageType::LBM:
            return {IMG_LoadLBM_RW(sdl_stream_io), &FreeSDLImage};
        case SDLImageType::PCX:
            return {IMG_LoadPCX_RW(sdl_stream_io), &FreeSDLImage};
        case SDLImageType::PNM:
            return {IMG_LoadPNM_RW(sdl_stream_io), &FreeSDLImage};
        case SDLImageType::TIF:
            return {IMG_LoadTIF_RW(sdl_stream_io), &FreeSDLImage};
        case SDLImageType::XCF:
            return {IMG_LoadXCF_RW(sdl_stream_io), &FreeSDLImage};
        case SDLImageType::XPM:
            return {IMG_LoadXPM_RW(sdl_stream_io), &FreeSDLImage};
        case SDLImageType::XV:
            return {IMG_LoadXV_RW(sdl_stream_io), &FreeSDLImage};
        case SDLImageType::WEBP:
            return {IMG_LoadWEBP_RW(sdl_stream_io), &FreeSDLImage};
        case SDLImageType::Unknown:
            return {nullptr, &FreeSDLImage};
    }
    return {nullptr, &FreeSDLImage};
}


}
