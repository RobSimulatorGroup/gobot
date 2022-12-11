/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * This file is created by Qiqi Wu, 22-11-20
*/

#include "gobot/core/ref_counted.hpp"

namespace gobot {

class Resource;

class ResourceFormatLoader : public RefCounted {
    GOBCLASS(ResourceFormatLoader, RefCounted);
public:
    enum class CacheMode {
        Ignore, // Resource and subresources do not use path cache, no path is set into resource.
        Reuse, // Resource and subresources use patch cache, reuse existing loaded resources instead of loading from disk when available.
        Replace // Resource and subresource use path cache, but replace existing loaded resources when available with information from disk.
    };

};


class ResourceLoader {

public:
    static Ref<Resource> Load(const String &path,
                              const String &type_hint = "",
                              ResourceFormatLoader::CacheMode cache_mode = ResourceFormatLoader::CacheMode::Reuse);

    static bool Exists(const String &path, const String &type_hint = "");


private:


};

}