/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * This file is created by Qiqi Wu, 22-11-20
*/

#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/io/resource.hpp"
#include "gobot/core/registration.hpp"

namespace gobot {


Ref<Resource> ResourceLoader::Load(const String &path,
                                   const String &type_hint,
                                   ResourceFormatLoader::CacheMode cache_mode) {


}

bool ResourceLoader::Exists(const String &path, const String &type_hint) {


}


}

GOBOT_REGISTRATION {

    gobot::Class_<gobot::ResourceLoader>("ResourceLoader");

};