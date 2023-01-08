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
#include <QFile>

namespace gobot {

class GOBOT_API ResourceFormatLoaderScene : public ResourceFormatLoader {
public:
    static ResourceFormatLoaderScene& GetSingleton();

    Ref<Resource> Load(const String &p_path,
                       const String &p_original_path = "",
                       CacheMode p_cache_mode = CacheMode::Reuse) override;

    void GetRecognizedExtensionsForType(const String& type, std::vector<String>* extensions) const override;

    void GetRecognizedExtensions(std::vector<String> *extensions) const override;

    bool HandlesType(const String& type) const override;

private:
    ResourceFormatLoaderScene();

};

class GOBOT_API ResourceLoaderSceneInstance {
public:
    ResourceLoaderSceneInstance();

    bool LoadResource();

    [[nodiscard]] Ref<Resource> GetResource() const;
private:
    friend ResourceFormatLoaderScene;

    Ref<Resource> resource_{nullptr};
    String file_context_;
    String local_path_;
    bool is_scene_{false};
    String res_type_;
    ResourceFormatLoaderScene::CacheMode cache_mode_;

    struct ExtResource {
        Ref<Resource> cache;
        String path;
        String type;
    };

    std::unordered_map<String, ExtResource> ext_resources_;
    std::unordered_map<String, Ref<Resource>> sub_resources_;
};


class GOBOT_API ResourceFormatSaverSceneInstance {
public:
    bool Save(const String &path, const Ref<Resource> &resource, ResourceSaverFlags flags = ResourceSaverFlags::None);

private:
    friend class VariantSerializer;

    void FindResources(const Variant &variant, bool main = false);

    Ref<PackedScene> packed_scene_;
    String local_path_;
    std::unordered_map<Ref<Resource>, Uuid> external_resources_;
    std::unordered_map<Ref<Resource>, Uuid> internal_resources_;
    std::vector<Ref<Resource>> saved_resources_;
    std::unordered_set<Ref<Resource>> resource_set_;
};

class GOBOT_API ResourceFormatSaverScene : public ResourceFormatSaver {
public:
    bool Save(const Ref<Resource> &resource, const String &path, ResourceSaverFlags flags = ResourceSaverFlags::None) override;

    void GetRecognizedExtensions(const Ref<Resource> &resource, std::vector<String>* extensions) const override;

    bool Recognize(const Ref<Resource> &resource) const override;

    static ResourceFormatSaverScene& GetSingleton();

private:
    ResourceFormatSaverScene();
};

}