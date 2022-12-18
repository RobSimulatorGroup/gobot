/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-11
*/

#pragma once

#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/io/resource_saver.hpp"

namespace gobot {

class ResourceFormatLoaderJson : public ResourceFormatLoader {
public:
//    virtual Ref<Resource> load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE);
//    virtual void get_recognized_extensions(List<String> *p_extensions) const;
    virtual bool handles_type(const String &p_type) const;
    virtual String get_resource_type(const String &p_path) const;
};


class ResourceFormatSaverJson : public ResourceFormatSaver {
public:
    virtual bool save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags = 0);
//    virtual void get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions) const;
    virtual bool recognize(const Ref<Resource> &p_resource) const;
};


}