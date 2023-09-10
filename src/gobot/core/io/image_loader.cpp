/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-15
*/


#include "gobot/core/io/image_loader.hpp"
#include <fstream>
#include "gobot/core/config/project_setting.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/core/string_utils.hpp"


namespace gobot {

bool ImageFormatLoader::Recognize(const std::string& extension) const {
    std::vector<std::string> extensions;
    GetRecognizedExtensions(&extensions);

    if (std::ranges::any_of(extensions, [&](auto const& ext){ return ToLower(ext) == ToLower(extension); })) {
        return true;
    }

    return false;
}

std::vector<Ref<ImageFormatLoader>> ImageLoader::s_loaders;

Ref<Image> ImageLoader::LoadImage(const std::string& path,
                                  LoaderFlags flags,
                                  float scale) {

    std::string global_path = ProjectSettings::GetInstance()->GlobalizePath(ValidateLocalPath(path));

    ERR_FAIL_COND_V_MSG(!std::filesystem::exists(global_path), {}, fmt::format("Cannot open file: {}.", path));
    std::ifstream instream(global_path, std::ios::in | std::ios::binary);
    std::vector<uint8_t> byte_array((std::istreambuf_iterator<char>(instream)), std::istreambuf_iterator<char>());

    std::string extension = GetFileExtension(global_path);

    for (int i = 0; i < s_loaders.size(); i++) {
        if (!s_loaders[i]->Recognize(extension)) {
            continue;
        }
        Ref<Image> image = s_loaders[i]->LoadImage(byte_array, flags, scale);
        if (!image) {
            LOG_ERROR("Error loading image: " + path);
        }
        return image;
    }

    return {};
}

void ImageLoader::GetRecognizedExtensions(std::vector<std::string>* extensions) {
    for (int i = 0; i < s_loaders.size(); i++) {
        s_loaders[i]->GetRecognizedExtensions(extensions);
    }
}

Ref<ImageFormatLoader> ImageLoader::Recognize(const std::string& extension) {
    for (int i = 0; i < s_loaders.size(); i++) {
        if (s_loaders[i]->Recognize(extension)) {
            return s_loaders[i];
        }
    }

    return nullptr;
}

void ImageLoader::AddImageFormatLoader(Ref<ImageFormatLoader> loader) {
    s_loaders.push_back(loader);
}

void ImageLoader::RemoveImageFormatLoader(Ref<ImageFormatLoader> loader) {
    auto it = std::find(s_loaders.begin(), s_loaders.end(), loader);

    ERR_FAIL_COND_MSG(it == s_loaders.end(), "The image loader to be removed is not find");

    s_loaders.erase(it);
}

void ImageLoader::Cleanup() {
    s_loaders.clear();
}


}