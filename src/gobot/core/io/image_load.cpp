/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-13
*/

#include "gobot/core/io/image_load.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/log.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/drivers/sdl/sdl_image.hpp"
#include <QFile>

namespace gobot {

ResourceFormatLoaderSDLImage* ResourceFormatLoaderSDLImage::s_singleton = nullptr;

ResourceFormatLoaderSDLImage::ResourceFormatLoaderSDLImage()
{
    s_singleton = this;
}

ResourceFormatLoaderSDLImage* ResourceFormatLoaderSDLImage::GetInstance()
{
    return s_singleton;
}


Ref<Resource> ResourceFormatLoaderSDLImage::Load(const String &path, CacheMode cache_mode)
{
    auto global_path = ProjectSettings::GetInstance()->GlobalizePath(path);
    ERR_FAIL_COND_V_MSG(!QFile::exists(global_path), {}, fmt::format("File: {} doesn't exists.", path));

    SDLStreamIO *io = SDL_RWFromFile(global_path.toStdString().c_str(), "rb");
    ERR_FAIL_COND_V_MSG(io == nullptr, {}, fmt::format("Can not open {}", path));

    auto sdl_image_type = SDLImageHandle::GetSDLImageType(io);
    ERR_FAIL_COND_V_MSG(sdl_image_type == SDLImageType::Unknown, {}, "");

    Ref<Image> image = New<Image>();
    image->SetSDLImage(SDLImageHandle::LoadSDLImage(sdl_image_type, io));

    return image;
}

void ResourceFormatLoaderSDLImage::GetRecognizedExtensions(std::vector<String> *extensions) const {
    extensions->reserve(extensions->size() + SDLImageHandle::s_sdl_image_types.size());
    extensions->insert(extensions->end(),
                       SDLImageHandle::s_sdl_image_types.begin(), SDLImageHandle::s_sdl_image_types.end());
}

bool ResourceFormatLoaderSDLImage::HandlesType(const String& type) const {
    return type == "Image";
}

}
