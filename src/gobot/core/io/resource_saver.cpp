/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * This file is created by Qiqi Wu, 22-11-20
*/

#include "gobot/core/io/resource_saver.hpp"

#include <utility>
#include "gobot/core/string_utils.hpp"
#include "gobot/log.hpp"

namespace gobot {

bool ResourceSaver::Save(const Ref<Resource>& resource, const String& target_path, ResourceSaverFlags flags) {
    String path = target_path;
    if (path.isEmpty()) {
        path = resource->GetPath();
    }
    if (path.isEmpty()) {
        LOG_ERROR("Can't save resource to empty path. Provide non-empty path or a Resource with non-empty resource_path.");
        return false;
    }

    auto extension = GetFileExtension(path);

    for (auto & s_saver : s_savers) {
        if (!s_saver->Recognize(resource)) {
            continue;
        }

        if (!s_saver->RecognizePath(resource, path)) {
            continue;
        }

        String old_path = resource->GetPath();

        // TODO(wqq)
        String local_path = path;

        Ref<Resource> rwcopy = resource;
        if (flags & ResourceSaverFlags::ChangePath) {
            rwcopy->SetPath(local_path);
        }

        if (s_saver->Save(resource, path)) {
            if (flags & ResourceSaverFlags::ChangePath) {
                rwcopy->SetPath(old_path);
            }
            if (resource_saved_callback && path.startsWith("res://")) {
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

void ResourceSaver::GetRecognizedExtensions(const Ref<Resource> &resource, std::vector<String>* extensions) {
    for (auto & s_saver : s_savers) {
        s_saver->GetRecognizedExtensions(resource, extensions);
    }
}

void ResourceSaver::AddResourceFormatSaver(Ref<ResourceFormatSaver> format_saver, bool at_front) {
    if (format_saver.is_valid()) {
        LOG_ERROR("It's not a reference to a valid ResourceFormatSaver object.");
        return;
    }

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