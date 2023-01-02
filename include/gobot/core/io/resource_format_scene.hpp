/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-11
*/

#pragma once

#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/io/resource_saver.hpp"
#include "gobot/scene/resources/packed_scene.hpp"

namespace gobot {

class ResourceFormatLoaderScene : public ResourceFormatLoader {
public:
//    virtual Ref<Resource> load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE);
//    virtual void get_recognized_extensions(List<String> *p_extensions) const;
    virtual bool handles_type(const String &p_type) const;
    virtual String get_resource_type(const String &p_path) const;
};


class ResourceFormatSaverSceneInstance {
public:
    bool Save(const String &path, const Ref<Resource> &resource, ResourceSaverFlags flags = ResourceSaverFlags::None);

private:
    void FindResources(const Variant &variant, bool main = false);

    Ref<PackedScene> packed_scene_;
    String local_path_;
    std::unordered_map<Ref<Resource>, Uuid> external_resources_;
    std::unordered_map<Ref<Resource>, Uuid> internal_resources_;
    std::vector<Ref<Resource>> saved_resources_;
    std::unordered_set<Ref<Resource>> resource_set_;
};

class ResourceFormatSaverScene : public ResourceFormatSaver {
public:
    bool Save(const Ref<Resource> &resource, const String &path, ResourceSaverFlags flags = ResourceSaverFlags::None) override;

    void GetRecognizedExtensions(const Ref<Resource> &resource, std::vector<String>* extensions) const override;

    bool Recognize(const Ref<Resource> &resource) const override;
};

}