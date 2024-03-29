/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * This file is created by Qiqi Wu, 22-11-20
*/

#include "gobot/core/io/resource_saver.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/log.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/error_macros.hpp"
#include <utility>


namespace gobot {

bool ResourceFormatSaver::RecognizePath(const Ref<Resource> &resource, const std::string &path) const {
    auto extension = GetFileExtension(path);

    std::vector<std::string> extensions;
    GetRecognizedExtensions(resource, &extensions);

    if (std::ranges::any_of(extensions, [&](auto const& ext){ return ToLower(ext) == ToLower(extension); })) {
        return true;
    }

    return false;
}

bool ResourceFormatSaver::Recognize(const Ref<Resource> &resource) const {
    return false;
}

std::deque<Ref<ResourceFormatSaver>> ResourceSaver::s_savers;
ResourceSaver::ResourceSavedCallback ResourceSaver::resource_saved_callback;
bool ResourceSaver::s_timestamp_on_save = false;

bool ResourceSaver::Save(const Ref<Resource>& resource, const std::string& target_path, ResourceSaverFlags flags) {
    std::string path = target_path;
    if (path.empty()) {
        path = resource->GetPath();
    }

    ERR_FAIL_COND_V_MSG(path.empty(), false,
                        "Can't save resource to empty path. Provide non-empty path or a Resource with non-empty resource_path.");

    for (auto & s_saver : s_savers) {
        if (!s_saver->Recognize(resource)) {
            continue;
        }

        if (!s_saver->RecognizePath(resource, path)) {
            continue;
        }

        std::string old_path = resource->GetPath();

        std::string local_path = ProjectSettings::GetInstance()->LocalizePath(path);

        USING_ENUM_BITWISE_OPERATORS;

        Resource* rwcopy = resource.Get();
        if ((bool)(flags & ResourceSaverFlags::ChangePath)) {
            rwcopy->SetPath(local_path);
        }

        if (s_saver->Save(resource, path)) {
            if ((bool)(flags & ResourceSaverFlags::ChangePath)) {
                rwcopy->SetPath(old_path);
            }
            if (resource_saved_callback && path.starts_with("res://")) {
                resource_saved_callback(resource, path);
            }

            return true;
        }
    }

    return false;
}

void ResourceSaver::SetSaveCallback(ResourceSavedCallback callback) {
    resource_saved_callback = std::move(callback);
}

void ResourceSaver::GetRecognizedExtensions(const Ref<Resource> &resource, std::vector<std::string>* extensions) {
    for (auto & s_saver : s_savers) {
        s_saver->GetRecognizedExtensions(resource, extensions);
    }
}

void ResourceSaver::AddResourceFormatSaver(const Ref<ResourceFormatSaver>& format_saver, bool at_front) {
    ERR_FAIL_COND_MSG(!format_saver.IsValid(), "It's not a reference to a valid ResourceFormatSaver object.");

    if (at_front) {
        s_savers.push_front(format_saver);
    } else {
        s_savers.push_back(format_saver);
    }
}

void ResourceSaver::RemoveResourceFormatSaver(const Ref<ResourceFormatSaver>& format_saver) {
    auto it = std::find(s_savers.begin(), s_savers.end(), format_saver);

    if (it == s_savers.end()) {
        LOG_ERROR("The format saver is not");
        return;
    }

    s_savers.erase(it);
}

}