/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-11
*/

#include "gobot/core/io/resource_format_scene.hpp"
#include <fstream>
#include "gobot/scene/resources/packed_scene.hpp"
#include "gobot/core/config/project_setting.hpp"
#include "gobot/log.hpp"
#include "gobot/core/types.hpp"
#include "gobot/type_categroy.hpp"
#include "gobot/core/io/variant_serializer.hpp"
#include "gobot/scene/resources/packed_scene.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/error_macros.hpp"

#define FORMAT_VERSION 1

namespace gobot {


ResourceFormatLoaderSceneInstance::ResourceFormatLoaderSceneInstance() {

}

bool ResourceFormatLoaderSceneInstance::LoadResource() {
    Json json = Json::parse(file_context_);
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
        is_scene_ = meta_type == "SCENE";
    } else {
        LOG_ERROR("The json: {} must contains __META_TYPE__", json);
        return false;
    }
    if (json.contains("__TYPE__")) {
        res_type_ = json["__TYPE__"];
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

                std::string path = ext_res["__PATH__"];
                std::string type = ext_res["__TYPE__"];
                std::string id = ext_res["__ID__"];

                if (path.find("://") == std::string::npos && IsRelativePath(path)) {
                    // path is relative to file being loaded, so convert to a resource path
                    path = ProjectSettings::GetInstance()->LocalizePath(PathJoin(GetBaseDir(local_path_), path));
                }

                Ref<Resource> res = ResourceLoader::Load(path, type);
                if (!res) {
                    LOG_ERROR("Cannot load type:{} from path:{}", type, path);
                    return false;
                } else {
                    res->SetUniqueId(id);
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

                std::string type = std::string(sub_res["__TYPE__"]);
                std::string id = std::string(sub_res["__ID__"]);

                // local resource's id is local_path + "::" + id
                std::string path = local_path_ + "::" + id;

                Ref<Resource> res;
                bool do_assign = false;
                if (cache_mode_ == ResourceFormatLoader::CacheMode::Replace && ResourceCache::Has(path)) {
                    //reuse existing
                    Ref<Resource> cache = ResourceCache::GetRef(path);
                    if (cache.IsValid() && cache->GetClassStringName().data() == type) {
                        res = cache;
                        res->ResetState();
                        do_assign = true;
                    }
                }

                if (!res) {
                    Ref<Resource> cache = ResourceCache::GetRef(path);
                    if (cache_mode_ != ResourceFormatLoader::CacheMode::Ignore && cache.IsValid()) { //only if it doesn't exist
                        //cached, do not assign
                        res = cache;
                    } else {
                        //create
                        Variant new_obj = Type::get_by_name(type).create();
                        bool success{false};
                        auto* r = new_obj.convert<Resource*>(&success);

                        if (!success) {
                            LOG_ERROR("Can't create sub resource of type, because not a Resource: {}", type);
                            return false;
                        }

                        res = Ref<Resource>(r);
                        do_assign = true;

                        for (const auto&[key, value] : sub_res.items()) {
                            if (key.starts_with("__")) {
                                continue;
                            }
                            auto property_value = res->Get(key.c_str());
                            if (!property_value.is_valid()) {
                                continue;
                            }

                            if (!VariantSerializer::JsonToVariant(property_value, value, this)) {
                                LOG_ERROR("Convert Json:{} to Variant({})", value.dump(4), property_value.get_type().get_name().data());
                                continue;
                            }

                            if (!property_value.convert(res->GetPropertyType(key.c_str()))) {
                                LOG_ERROR("Cannot convert source type: {} to target type: {}",
                                          property_value.get_type().get_name().data(),
                                          res->GetPropertyType(key.c_str()).get_name().data());
                                continue;
                            }

                            if (!res->Set(key.c_str(), property_value)) {
                                LOG_ERROR("Convert set key:{} to resource", key.c_str());
                                continue;
                            }
                        }
                    }
                }

                sub_resources_[id] = res; //always assign int resources
                if (do_assign) {
                    if (cache_mode_ == ResourceFormatLoader::CacheMode::Ignore) {
                        res->SetPath(path);
                    } else {
                        res->SetPath(path, cache_mode_ == ResourceFormatLoader::CacheMode::Replace);
                        res->SetUniqueId(id);
                    }
                }

            }
        } else {
            LOG_ERROR("__SUB_RESOURCES__ must array");
            return {};
        }
    }

    if (json.contains("__RESOURCE__")) {
        if (is_scene_) {
            LOG_ERROR("__RESOURCE__ and __META_TYPE__: SCENE are conflicted");
            return false;
        }

        Ref<Resource> cache = ResourceCache::GetRef(local_path_);
        if (cache_mode_ == ResourceFormatLoader::CacheMode::Replace && cache.IsValid() &&
                cache->GetClassStringName().data() == res_type_) {
            cache->ResetState();
            resource_ = cache;
        }

        if (!resource_.IsValid()) {

            Variant new_obj = Type::get_by_name(res_type_).create();
            bool success{false};
            auto *r = new_obj.convert<Resource*>(&success);
            if (!success) {
                LOG_ERROR("Can't create sub resource of type, because not a resource: {}", res_type_);
                return false;
            }

            resource_ = Ref<Resource>(r);
        }

        for (const auto&[key, value] : json["__RESOURCE__"].items()) {
            auto property_value = resource_->Get(key.c_str());
            if (!property_value.is_valid()) {
                continue;
            }

            if (!VariantSerializer::JsonToVariant(property_value, value, this)) {
                LOG_ERROR("Convert Json:{} to Variant({})", value.dump(4), property_value.get_type().get_name().data());
                continue;
            }

            if (!property_value.convert(resource_->GetPropertyType(key.c_str()))) {
                LOG_ERROR("Cannot convert source type: {} to target type: {}",
                          property_value.get_type().get_name().data(),
                          resource_->GetPropertyType(key.c_str()).get_name().data());
                continue;
            }

            if (!resource_->Set(key.c_str(), property_value)) {
                LOG_ERROR("Convert set key:{} to resource", key.c_str());
                continue;
            }
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

Ref<Resource> ResourceFormatLoaderScene::Load(const std::string &path,
                                              const std::string &original_path,
                                              CacheMode cache_mode) {
    // Ignore original_path because path and original_path are same for scene.
    auto global_path = ProjectSettings::GetInstance()->GlobalizePath(path);

    ERR_FAIL_COND_V_MSG(!std::filesystem::exists(global_path), {}, fmt::format("Cannot open file: {}.", path));

    std::ifstream ifstream(global_path);
    std::string str((std::istreambuf_iterator<char>(ifstream)),
                    std::istreambuf_iterator<char>());

    ResourceFormatLoaderSceneInstance loader;
    loader.file_context_ = str;
    loader.cache_mode_ = cache_mode;
    loader.local_path_ = ProjectSettings::GetInstance()->LocalizePath(path);

    if (loader.LoadResource()) {
        return loader.GetResource();
    }

    return {};
}

ResourceFormatLoaderScene* ResourceFormatLoaderScene::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize ResourceFormatLoaderScene");
    return s_singleton;
}

void ResourceFormatLoaderScene::GetRecognizedExtensionsForType(const std::string& type, std::vector<std::string>* extensions) const {
    if (type.empty()) {
        GetRecognizedExtensions(extensions);
        return;
    }

    auto type_class = Type::get_by_name(type);
    auto packed_scene_type = Type::get<PackedScene>();
    if (packed_scene_type == type_class) {
        extensions->push_back("jscn");
    }

    if (type_class != packed_scene_type) {
        extensions->push_back("jres");
    }
}


void ResourceFormatLoaderScene::GetRecognizedExtensions(std::vector<std::string> *extensions) const {
    extensions->push_back("jscn");
    extensions->push_back("jres");
}

bool ResourceFormatLoaderScene::HandlesType(const std::string& type) const {
    return true;
}


///////////////////////////////////////////////////////////////////

bool ResourceFormatSaverSceneInstance::Save(const std::string &path, const Ref<Resource> &resource, ResourceSaverFlags flags)
{
    if (path.ends_with(".jscn")) {
        packed_scene_ = gobot::dynamic_pointer_cast<PackedScene>(resource);
    }

    auto global_path = ProjectSettings::GetInstance()->GlobalizePath(path);
    auto base_dir = GetBaseDir(global_path);
    if (!std::filesystem::exists(base_dir))
        std::filesystem::create_directories(base_dir); // You can check the success if needed


    USING_ENUM_BITWISE_OPERATORS;
    takeover_paths_ = static_cast<bool>(flags & ResourceSaverFlags::ReplaceSubResourcePaths);

    local_path_ = ProjectSettings::GetInstance()->LocalizePath(path);

    // Save resources.
    FindResources(resource, true);

    if (packed_scene_.IsValid()) {
        // Add instances to external resources if saving a packed scene.
        for (std::size_t i = 0; i < packed_scene_->GetState()->GetNodeCount(); i++) {
            Ref<PackedScene> instance = packed_scene_->GetState()->GetNodeInstance(i);
            if (instance.IsValid() && !external_resources_.contains(instance)) {
                external_resources_[instance] = Resource::GenerateResourceUniqueId();
            }
        }
    }

    Json root;
    root["__VERSION__"] = FORMAT_VERSION;
    root["__META_TYPE__"] = packed_scene_.IsValid() ? "SCENE" : "RESOURCE";

    root["__EXT_RESOURCES__"] = Json::array();
    for (const auto& [res, id]: external_resources_) {
        Json ext_res;
        ext_res["__TYPE__"] = res->GetClassStringName();
        ext_res["__PATH__"] = res->GetPath();
        ext_res["__ID__"] = id;
        root["__EXT_RESOURCES__"].push_back(ext_res);
    }

    std::unordered_set<std::string> used_unique_ids;

    for (auto& res : saved_resources_) {
        if (res != saved_resources_.back()  && res->IsBuiltIn()) {
            if (!res->GetUniqueId().empty()) {
                if (used_unique_ids.contains(res->GetUniqueId())) {
                    res->SetUniqueId(""); // Repeated.
                } else {
                    used_unique_ids.emplace(res->GetUniqueId());
                }
            }
        }
    }

    for (const auto& saved_resource: saved_resources_) {
        if (!resource_set_.contains(saved_resource)) {
            continue;
        }
        bool main = (saved_resource == saved_resources_.back());
        if (main && packed_scene_.IsValid()) {
            break; // Save as a scene.
        }
        Json resource_data_json;
        auto type = Object::GetDerivedTypeByInstance(saved_resource);

        for (auto& prop : type.get_properties()) {
            if (prop.is_readonly()) {
                continue;
            }
            PropertyInfo property_info;
            auto property_metadata = prop.get_metadata(PROPERTY_INFO_KEY);
            if (property_metadata.is_valid()) {
                property_info = property_metadata.get_value<PropertyInfo>();
            }
            USING_ENUM_BITWISE_OPERATORS;
            if ((bool)(property_info.usage & PropertyUsageFlags::Storage)) {
                Variant value = saved_resource->Get(prop.get_name().data());
                resource_data_json[prop.get_name().data()] = VariantSerializer::VariantToJson(value, this);;
            }
        }

        if (main) {
            root["__RESOURCE__"] = resource_data_json;
            root["__TYPE__"] = type.get_name().data();
        } else {
            if (!root.contains("__SUB_RESOURCES__")) {
                root["__SUB_RESOURCES__"] = Json::array();
            }
            auto class_name = type.get_name();
            resource_data_json["__TYPE__"] = class_name;
            if (saved_resource->GetUniqueId().empty()) {
                std::string new_id;
                while (true) {
                    new_id = std::string(class_name.data()) + "_" + Resource::GenerateResourceUniqueId();

                    if (!used_unique_ids.contains(new_id)) {
                        break;
                    }
                }

                saved_resource->SetUniqueId(new_id);
                used_unique_ids.insert(new_id);
            }
            if (takeover_paths_) {
                saved_resource->SetPath(path + "::" + saved_resource->GetUniqueId(), true);
            }
            internal_resources_[saved_resource] = saved_resource->GetUniqueId();

            resource_data_json["__ID__"] = saved_resource->GetUniqueId();
            root["__SUB_RESOURCES__"].emplace_back(resource_data_json);
        }
    }

    std::ofstream out_file(global_path);
    out_file << root.dump(4);
    out_file.close();
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
        if (!res.IsValid() || external_resources_.contains(res)) {
            return;
        }
        // built in means internal_resources_
        if (!main && !res->IsBuiltIn()) {
            if (res->GetPath() == local_path_) {
                LOG_ERROR("Circular reference to resource being saved found: {}.", local_path_);
                return;
            }

            external_resources_[res] = std::to_string(external_resources_.size() + 1) + "_" + Resource::GenerateResourceUniqueId();
            return;
        }

        if (resource_set_.contains(res)) {
            return;
        }


        for (auto& prop : Object::GetDerivedTypeByInstance(variant).get_properties()) {
            if (prop.is_readonly()) {
                continue;
            }
            PropertyInfo property_info;
            auto meta_data = prop.get_metadata(PROPERTY_INFO_KEY);
            if (meta_data.is_valid()) {
                property_info = meta_data.get_value<PropertyInfo>();
            }
            USING_ENUM_BITWISE_OPERATORS;
            if ((bool)(property_info.usage & PropertyUsageFlags::Storage)) {
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
            PropertyInfo property_info;
            auto meta_data = prop.get_metadata(PROPERTY_INFO_KEY);
            if (meta_data.is_valid()) {
                property_info = meta_data.get_value<PropertyInfo>();
            }
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

ResourceFormatSaverScene* ResourceFormatSaverScene::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize ResourceFormatSaverScene");
    return s_singleton;
}

bool ResourceFormatSaverScene::Save(const Ref<Resource> &resource, const std::string &path, ResourceSaverFlags flags) {
    if (path.ends_with(".jscn") && !gobot::dynamic_pointer_cast<PackedScene>(resource)) {
        return false;
    }

    ResourceFormatSaverSceneInstance saver;
    return saver.Save(path, resource, flags);
}

void ResourceFormatSaverScene::GetRecognizedExtensions(const Ref<Resource> &resource,
                                                       std::vector<std::string>* extensions) const {
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