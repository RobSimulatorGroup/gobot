/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-11
*/


#include "gobot/core/io/resource_format_scene.hpp"
#include "gobot/scene/resources/packed_scene.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/log.hpp"
#include "gobot/core/types.hpp"
#include "gobot/type_categroy.hpp"
#include "gobot/core/io/variant_serializer.hpp"
#include "gobot/scene/resources/packed_scene.hpp"
#include "gobot/core/string_utils.hpp"

#include <QTextStream>
#include <QFile>

#define FORMAT_VERSION 1

namespace gobot {


ResourceFormatLoaderSceneInstance::ResourceFormatLoaderSceneInstance() {

}

bool ResourceFormatLoaderSceneInstance::LoadResource() {
    Json json = Json::parse(file_context_.toStdString());
    if (json.contains("__VERSION__")) {
        float version = json["__VERSION__"];
        if (version > FORMAT_VERSION) {
            LOG_ERROR("__VERSION__ must be {}", FORMAT_VERSION);
            return false;
        }
    } else {
        LOG_ERROR("The json: {} must contains __VERSION__", json);
        return false;
    }
    if (json.contains("__META_TYPE__")) {
        std::string meta_type = json["__META_TYPE__"];
        if (meta_type != "SCENE" && meta_type != "RESOURCE") {
            LOG_ERROR("__META_TYPE__ must SCENE or RESOURCE");
            return false;
        }
    } else {
        LOG_ERROR("The json: {} must contains __META_TYPE__", json);
        return false;
    }

    if (json.contains("__EXT_RESOURCES__")) {
        Json ext_resources = json["__EXT_RESOURCES__"];
        if (ext_resources.is_array()) {
            for (const auto& ext_res: ext_resources) {
                if (!ext_res.contains("__PATH__")) {
                    LOG_ERROR("Missing '__PATH__' in external resource");
                    return false;
                }

                if (!ext_res.contains("__TYPE__")) {
                    LOG_ERROR("Missing '__TYPE__' in external resource");
                    return false;
                }

                if (!ext_res.contains("__ID__")) {
                    LOG_ERROR("Missing '__ID__' in external resource");
                    return false;
                }

                auto path = String::fromStdString(ext_res["__PATH__"]);
                auto type = String::fromStdString(ext_res["__TYPE__"]);
                auto id = String::fromStdString(ext_res["__ID__"]);

                if (!path.contains("://") && IsRelativePath(path)) {
                    // path is relative to file being loaded, so convert to a resource path
                    path = ProjectSettings::GetSingleton()->LocalizePath(PathJoin(GetBaseDir(local_path_), path));
                }

                Ref<Resource> res = ResourceLoader::Load(path, type);
                if (!res) {
                    LOG_ERROR("");
                    return false;
                } else {
				    res->SetResourceUuid(id);
                }

                ext_resources_[id] = res;
            }
        }
    }

    if (json.contains("__SUB_RESOURCES__")) {
        Json sub_resources = json["__SUB_RESOURCES__"];
        if (sub_resources.is_array()) {
            for (const auto& sub_res: sub_resources) {
                if (!sub_res.contains("__TYPE__")) {
                    LOG_ERROR("Missing '__TYPE__' in external resource");
                    return false;
                }

                if (!sub_res.contains("__ID__")) {
                    LOG_ERROR("Missing '__TYPE__' in external resource");
                    return false;
                }

                String type = String::fromStdString(to_string(sub_res["__TYPE__"]));
                String id = String::fromStdString(to_string(sub_res["__ID__"]));

                String path = local_path_ + "::" + id;

                Ref<Resource> res;
                bool do_assign = false;
                if (cache_mode_ == ResourceFormatLoader::CacheMode::Replace && ResourceCache::Has(path)) {
                    //reuse existing
                    Ref<Resource> cache = ResourceCache::GetRef(path);
                    if (cache.is_valid() && cache->GetClassName().data() == type) {
                        res = cache;
                        res->ResetState();
                        do_assign = true;
                    }
                }

                if (!res) {
                    Ref<Resource> cache = ResourceCache::GetRef(path);
                    if (cache_mode_ != ResourceFormatLoader::CacheMode::Ignore && cache.is_valid()) { //only if it doesn't exist
                        //cached, do not assign
                        res = cache;
                    } else {
                        //create
                        Variant new_obj = Type::get_by_name(type.toStdString()).create();
                        if (!new_obj.can_convert<Object*>()) {
                            LOG_ERROR("");
                            return false;
                        }
                        auto* obj = new_obj.convert<Object*>();

                        auto *r = Object::PointerCastTo<Resource>(obj);
                        if (!r) {
                            LOG_ERROR("Can't create sub resource of type, because not a resource: {}", type);
                            return false;
                        }

                        res = Ref<Resource>(r);
                        do_assign = true;
                    }
                }

                sub_resources_[id] = res; //always assign int resources
                if (do_assign) {
                    if (cache_mode_ == ResourceFormatLoader::CacheMode::Ignore) {
                        res->SetPath(path);
                    } else {
                        res->SetPath(path, cache_mode_ == ResourceFormatLoader::CacheMode::Replace);
                        res->SetResourceUuid(id);
                    }
                }

            }
        } else {
            LOG_ERROR("");
            return {};
        }
    }

    if (json.contains("__RESOURCE__")) {
        if (is_scene_) {
            LOG_ERROR("");
            return false;
        }

        Ref<Resource> cache = ResourceCache::GetRef(local_path_);
        if (cache_mode_ == ResourceFormatLoader::CacheMode::Replace && cache.is_valid() && cache->GetClassName().data() == res_type_) {
            cache->ResetState();
            resource_ = cache;
        }

        if (!resource_.is_valid()) {

            Variant new_obj = Type::get_by_name(res_type_.toStdString()).create();
            if (!new_obj.can_convert<Object*>()) {
                LOG_ERROR("");
                return false;
            }
            auto* obj = new_obj.convert<Object*>();

            auto *r = Object::PointerCastTo<Resource>(obj);
            if (!r) {
                LOG_ERROR("Can't create sub resource of type, because not a resource: {}", res_type_);
                return false;
            }

            resource_ = Ref<Resource>(r);
        }

        for (const auto&[key, value] : json["__RESOURCE__"].items()) {
            auto value_type = resource_->GetPropertyType(key.c_str());
            auto variant = VariantSerializer::JsonToVariant(value_type, value);

            if (!variant.is_valid()) {
                LOG_ERROR("");
            }

            resource_->Set(key.c_str(), variant);
        }

        if (cache_mode_ != ResourceFormatLoader::CacheMode::Ignore) {
            if (!ResourceCache::Has(local_path_)) {
                resource_->SetPath(local_path_);
            }
        }

        return true;
    }

    if (json.contains("__NODES__")) {
        if (is_scene_) {
            LOG_ERROR("");
            return false;
        }


        return true;
    }

    return false;
}


Ref<Resource> ResourceFormatLoaderSceneInstance::GetResource() const {
    return resource_;
}

ResourceFormatLoaderScene* ResourceFormatLoaderScene::s_singleton = nullptr;

ResourceFormatLoaderScene::ResourceFormatLoaderScene() {
    s_singleton = this;
}

ResourceFormatLoaderScene::~ResourceFormatLoaderScene() {
    s_singleton = nullptr;
}

Ref<Resource> ResourceFormatLoaderScene::Load(const String &path,
                                              const String &original_path,
                                              CacheMode cache_mode) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        LOG_ERROR("Cannot open file: {}.", path);
        return {};
    }

    ResourceFormatLoaderSceneInstance loader;
    loader.file_context_ = file.readAll();
    loader.cache_mode_ = cache_mode;
    loader.local_path_ = ProjectSettings::GetSingleton()->LocalizePath(!original_path.isEmpty() ? original_path : path);

    if (loader.LoadResource()) {
        return loader.GetResource();
    }

    return {};
}

ResourceFormatLoaderScene* ResourceFormatLoaderScene::GetSingleton() {
    return s_singleton;
}

void ResourceFormatLoaderScene::GetRecognizedExtensionsForType(const String& type, std::vector<String>* extensions) const {
    if (type.isEmpty()) {
        GetRecognizedExtensions(extensions);
        return;
    }

    auto type_class = Type::get_by_name(type.toStdString());
    auto packed_scene_type = Type::get<PackedScene>();
    if (packed_scene_type == type_class) {
        extensions->push_back("jscn");
    }

    if (type_class != packed_scene_type) {
        extensions->push_back("jres");
    }
}


void ResourceFormatLoaderScene::GetRecognizedExtensions(std::vector<String> *extensions) const {
    extensions->push_back("jscn");
    extensions->push_back("jres");
}

bool ResourceFormatLoaderScene::HandlesType(const String& type) const {
    return true;
}


///////////////////////////////////////////////////////////////////

bool ResourceFormatSaverSceneInstance::Save(const String &path, const Ref<Resource> &resource, ResourceSaverFlags flags)
{
    if (path.endsWith(".jscn")) {
        packed_scene_ = gobot::dynamic_pointer_cast<PackedScene>(resource);
    }
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly)) {
        LOG_ERROR("Cannot save file: {}", path);
        return false;
    }

    local_path_ = ProjectSettings::GetSingleton()->LocalizePath(path);

    // Save resources.
    FindResources(resource, true);

    if (packed_scene_.is_valid()) {
        // Add instances to external resources if saving a packed scene.
        for (std::size_t i = 0; i < packed_scene_->GetState()->GetNodeCount(); i++) {
            Ref<PackedScene> instance = packed_scene_->GetState()->GetNodeInstance(i);
            if (instance.is_valid() && !external_resources_.contains(instance)) {
                external_resources_[instance] = Resource::GenerateUuid();
            }
        }
    }

    Json root;
    root["__VERSION__"] = FORMAT_VERSION;
    root["__META_TYPE__"] = packed_scene_.is_valid() ? "SCENE" : "RESOURCE";

    root["__EXT_RESOURCES__"] = Json::array();
    for (const auto& [res, uuid]: external_resources_) {
        Json ext_res;
        ext_res["__TYPE__"] = res->GetClassName();
        ext_res["__PATH__"] = res->GetPath().toStdString();
        ext_res["__ID__"] = res->GetResourceUuid().toString().toStdString();
        root["__EXT_RESOURCES__"].push_back(ext_res);
    }

    for (const auto& saved_resource: saved_resources_) {
        if (!resource_set_.contains(saved_resource)) {
            continue;
        }
        bool main = (saved_resource == saved_resources_.back());
        if (main && packed_scene_.is_valid()) {
            break; // Save as a scene.
        }
        Json resource_data_json;

        Variant variant = saved_resource;

        for (auto& prop : variant.get_type().get_properties()) {
            auto property_info = prop.get_metadata(PROPERTY_INFO_KEY).get_value<PropertyInfo>();
            USING_ENUM_BITWISE_OPERATORS;
            if ((bool)(property_info.usage & PropertyUsageFlags::Storage)) {
                Variant value = saved_resource->Get(prop.get_name().data());
                resource_data_json[prop.get_name().data()] = VariantSerializer::VariantToJson(value, this);
            }
        }

        if (main) {
            root["__RESOURCE__"] = resource_data_json;
            root["__TYPE__"] = variant.get_type().get_name().data();
            root["__ID__"] = saved_resource->GetResourceUuid().toString().toStdString();
        } else {
            if (!root.contains("__SUB_RESOURCES__")) {
                root["__SUB_RESOURCES__"] = Json::array();
            }
            resource_data_json["__TYPE__"] = variant.get_type().get_name().data();
            resource_data_json["__ID__"] = saved_resource->GetResourceUuid().toString().toStdString();
            root["__SUB_RESOURCES__"].emplace_back(resource_data_json);
        }
    }

    file.write(root.dump(4).c_str());
    return true;
}

void ResourceFormatSaverSceneInstance::FindResources(const Variant &variant, bool main) {
    auto type = variant.get_type();
    auto type_category = GetTypeCategory(type);

    if (type_category == TypeCategory::Ref) {
        bool success{false};
        auto res = variant.convert<Ref<Resource>>(&success);
        if (!success) {
            LOG_ERROR("Cannot convert variant: {} to Ref<Resource>", variant.get_type().get_name().data());
            return;
        }
        if (!res.is_valid() || external_resources_.contains(res)) {
            return;
        }
        if (!main && !res->IsBuiltIn()) {
            if (res->GetPath() == local_path_) {
                LOG_ERROR("Circular reference to resource being saved found: {}.", local_path_);
                return;
            }

            external_resources_[res] = Resource::GenerateUuid();
            return;
        }

        if (resource_set_.contains(res)) {
            return;
        }

        for (auto& prop : type.get_properties()) {
            auto property_info = prop.get_metadata(PROPERTY_INFO_KEY).get_value<PropertyInfo>();
            USING_ENUM_BITWISE_OPERATORS;
            if (!(bool)(property_info.usage & PropertyUsageFlags::Storage)) {
                Variant v = res->Get(prop.get_name().data());
                FindResources(v);
            }
        }

        resource_set_.insert(res); //saved after, so the children it needs are available when loaded
        saved_resources_.push_back(res);

    } else if (type_category == TypeCategory::Array) {
        auto view = variant.create_sequential_view();
        for (auto& item : view) {
            const Variant &v = item.extract_wrapped_value();
            FindResources(v);
        }
    } else if (type_category == TypeCategory::Dictionary) {
        auto view = variant.create_associative_view();
        for (auto& [key, value] : view) {
            const Variant &k = key.extract_wrapped_value();
            FindResources(k);
            const Variant &v = value.extract_wrapped_value();
            FindResources(v);
        }
    }  else if (type_category == TypeCategory::Compound) {
        for (auto &prop: type.get_properties()) {
            auto property_info = prop.get_metadata(PROPERTY_INFO_KEY).get_value<PropertyInfo>();
            USING_ENUM_BITWISE_OPERATORS;
            if (!(bool) (property_info.usage & PropertyUsageFlags::Storage)) {
                Variant v = prop.get_value(variant);
                FindResources(v);
            }
        }
    }

    // Other types won't handle.
}

ResourceFormatSaverScene* ResourceFormatSaverScene::s_singleton = nullptr;

ResourceFormatSaverScene::ResourceFormatSaverScene() {
    s_singleton = this;
}

ResourceFormatSaverScene::~ResourceFormatSaverScene() {
    s_singleton = nullptr;
}

ResourceFormatSaverScene* ResourceFormatSaverScene::GetSingleton() {
    return s_singleton;
}

bool ResourceFormatSaverScene::Save(const Ref<Resource> &resource, const String &path, ResourceSaverFlags flags) {
    if (path.endsWith(".jscn") && !gobot::dynamic_pointer_cast<PackedScene>(resource)) {
        return false;
    }

    ResourceFormatSaverSceneInstance saver;
    return saver.Save(path, resource, flags);
}

void ResourceFormatSaverScene::GetRecognizedExtensions(const Ref<Resource> &resource,
                                                       std::vector<String>* extensions) const {
    if (gobot::dynamic_pointer_cast<PackedScene>(resource)) {
        extensions->push_back("jscn"); // scene.
    } else {
        extensions->push_back("jres"); // resource.
    }
}

bool ResourceFormatSaverScene::Recognize(const Ref<Resource> &resource) const {
    return true;
}




}