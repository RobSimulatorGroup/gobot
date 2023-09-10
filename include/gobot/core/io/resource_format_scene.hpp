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


class GOBOT_EXPORT ResourceFormatLoaderSceneInstance {
public:
    ResourceFormatLoaderSceneInstance();

    bool LoadResource();

    [[nodiscard]] Ref<Resource> GetResource() const;
private:
    friend class ResourceFormatLoaderScene;
    friend class VariantSerializer;

    Ref<Resource> resource_{nullptr};
    std::string file_context_;
    std::string local_path_;
    bool is_scene_{false};
    std::string res_type_;
    ResourceFormatLoader::CacheMode cache_mode_{ResourceFormatLoader::CacheMode::Reuse};

    std::unordered_map<std::string, Ref<Resource>> ext_resources_;
    std::unordered_map<std::string, Ref<Resource>> sub_resources_;
};


class GOBOT_EXPORT ResourceFormatLoaderScene : public ResourceFormatLoader {
public:

    static ResourceFormatLoaderScene* s_singleton;

    ResourceFormatLoaderScene();

    ~ResourceFormatLoaderScene() override;

    static ResourceFormatLoaderScene* GetInstance();


    Ref<Resource> Load(const std::string &path,
                       const std::string &original_path = "",
                       CacheMode cache_mode = CacheMode::Reuse) override;

    void GetRecognizedExtensionsForType(const std::string& type, std::vector<std::string>* extensions) const override;

    void GetRecognizedExtensions(std::vector<std::string> *extensions) const override;

    [[nodiscard]] bool HandlesType(const std::string& type) const override;
};

class GOBOT_EXPORT ResourceFormatSaverSceneInstance {
public:
    bool Save(const std::string &path, const Ref<Resource> &resource, ResourceSaverFlags flags = ResourceSaverFlags::None);

private:
    friend class VariantSerializer;

    void FindResources(const Variant &variant, bool main = false);

    bool takeover_paths_ = false;
    Ref<PackedScene> packed_scene_;
    std::string local_path_;
    std::unordered_map<Ref<Resource>, std::string> external_resources_;
    std::unordered_map<Ref<Resource>, std::string> internal_resources_;
    std::vector<Ref<Resource>> saved_resources_;
    std::unordered_set<Ref<Resource>> resource_set_;
};

class GOBOT_EXPORT ResourceFormatSaverScene : public ResourceFormatSaver {
public:
    static ResourceFormatSaverScene* s_singleton;

    ResourceFormatSaverScene();

    ~ResourceFormatSaverScene() override;

    static ResourceFormatSaverScene* GetInstance();

    bool Save(const Ref<Resource> &resource, const std::string &path, ResourceSaverFlags flags = ResourceSaverFlags::None) override;

    void GetRecognizedExtensions(const Ref<Resource> &resource, std::vector<std::string>* extensions) const override;

    [[nodiscard]] bool Recognize(const Ref<Resource> &resource) const override;
};

}