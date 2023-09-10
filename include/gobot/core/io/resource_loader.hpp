/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * This file is created by Qiqi Wu, 22-11-20
*/

#pragma once

#include "gobot/core/ref_counted.hpp"

namespace gobot {

class Resource;

class GOBOT_EXPORT ResourceFormatLoader : public RefCounted {
    GOBCLASS(ResourceFormatLoader, RefCounted);
public:
    enum class CacheMode {
        Ignore, // Resource and subresources do not use path cache, no path is set into resource.
        Reuse, // Resource and subresources use path cache, reuse existing loaded resources instead of loading from disk when available.
        Replace // Resource and subresource use path cache, but replace existing loaded resources when available with information from disk.
    };

    virtual Ref<Resource> Load(const std::string &local_path,
                               const std::string &original_path = "",
                               CacheMode cache_mode = CacheMode::Reuse) = 0;

    [[nodiscard]] virtual bool HandlesType(const std::string &type) const = 0;

    [[nodiscard]] virtual bool RecognizePath(const std::string &path, const std::string &type_hint = "") const;

    virtual void GetRecognizedExtensions(std::vector<std::string> *extensions) const = 0;

    virtual void GetRecognizedExtensionsForType(const std::string &type, std::vector<std::string> *extensions) const;

    [[nodiscard]] virtual bool Exists(const std::string &path) const;

};


class GOBOT_EXPORT ResourceLoader {

public:
    static Ref<Resource> Load(const std::string &path,
                              const std::string &type_hint = "",
                              ResourceFormatLoader::CacheMode cache_mode = ResourceFormatLoader::CacheMode::Reuse);

    static bool Exists(const std::string &path, const std::string &type_hint = "");

    static void AddResourceFormatLoader(const Ref<ResourceFormatLoader>& format_loader, bool at_front = false);

    static void RemoveResourceFormatLoader(const Ref<ResourceFormatLoader>& format_loader);

    static void SetTimestampOnLoad(bool timestamp) { s_timestamp_on_load = timestamp; }

    static bool GetTimestampOnLoad() { return s_timestamp_on_load; }

private:

    static Ref<Resource> LoadImpl(const std::string &path,
                                  const std::string &type_hint,
                                  ResourceFormatLoader::CacheMode cache_mode);

    static std::deque<Ref<ResourceFormatLoader>> s_loaders;

    static bool s_timestamp_on_load;

};

}