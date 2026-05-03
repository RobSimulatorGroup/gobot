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
#include "gobot/scene/node.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/error_macros.hpp"

#define FORMAT_VERSION 2

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
                if (path.empty()) {
                    LOG_ERROR("External resource '{}' of type '{}' has an empty path in '{}'.",
                              id, type, local_path_);
                    return false;
                }

                if (path.find("://") == std::string::npos && IsRelativePath(path)) {
                    // path is relative to file being loaded, so convert to a resource path
                    path = ProjectSettings::GetInstance()->LocalizePath(PathJoin(GetBaseDir(local_path_), path));
                }

                ResourceFormatLoader::CacheMode external_cache_mode = ResourceFormatLoader::CacheMode::Reuse;
                if (type == "PackedScene" &&
                    (cache_mode_ == ResourceFormatLoader::CacheMode::Ignore ||
                     cache_mode_ == ResourceFormatLoader::CacheMode::Replace)) {
                    external_cache_mode = ResourceFormatLoader::CacheMode::Replace;
                }

                LOG_TRACE("Scene '{}' loading ExtResource id '{}' type '{}' path '{}' cache_mode {}.",
                          local_path_,
                          id,
                          type,
                          path,
                          static_cast<int>(external_cache_mode));
                Ref<Resource> res = ResourceLoader::Load(path, type, external_cache_mode);
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
                if (type == "PBRMaterial3D") {
                    auto albedo_it = sub_res.find("albedo");
                    if (albedo_it != sub_res.end()) {
                        LOG_TRACE("Scene '{}' loading SubResource id '{}' type '{}' albedo {}.",
                                  local_path_,
                                  id,
                                  type,
                                  albedo_it->dump());
                    }
                }

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
                    res->SetUniqueId(id);
                    if (cache_mode_ != ResourceFormatLoader::CacheMode::Ignore) {
                        res->SetPath(path, cache_mode_ == ResourceFormatLoader::CacheMode::Replace);
                    } else {
                        res->SetPathWithoutCache(path);
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
        if (!is_scene_) {
            LOG_ERROR("__NODES__ requires __META_TYPE__: SCENE");
            return false;
        }

        resource_ = MakeRef<PackedScene>();
        if (!LoadSceneNodes(json["__NODES__"])) {
            resource_ = {};
            return false;
        }

        if (cache_mode_ != ResourceFormatLoader::CacheMode::Ignore && !ResourceCache::Has(local_path_)) {
            resource_->SetPath(local_path_);
        }

        return true;
    }

    return false;
}

bool ResourceFormatLoaderSceneInstance::LoadSceneNodes(const Json& nodes_json) {
    if (!nodes_json.is_array()) {
        LOG_ERROR("__NODES__ must be an array");
        return false;
    }

    Ref<PackedScene> packed_scene = dynamic_pointer_cast<PackedScene>(resource_);
    if (!packed_scene.IsValid()) {
        LOG_ERROR("Cannot load scene nodes without a PackedScene resource");
        return false;
    }

    Ref<SceneState> state = packed_scene->GetState();
    state->Clear();

    for (const auto& node_json : nodes_json) {
        if (!node_json.is_object()) {
            LOG_ERROR("Scene node entry must be an object: {}", node_json.dump(4));
            return false;
        }

        SceneState::NodeData node_data;
        if (node_json.contains("type")) {
            node_data.type = node_json["type"].get<std::string>();
        } else if (node_json.contains("__TYPE__")) {
            node_data.type = node_json["__TYPE__"].get<std::string>();
        } else {
            LOG_ERROR("Scene node is missing type: {}", node_json.dump(4));
            return false;
        }

        if (node_json.contains("name")) {
            node_data.name = node_json["name"].get<std::string>();
        }

        if (node_json.contains("parent")) {
            node_data.parent = node_json["parent"].is_null() ? -1 : node_json["parent"].get<int>();
        } else if (node_json.contains("__PARENT__")) {
            node_data.parent = node_json["__PARENT__"].is_null() ? -1 : node_json["__PARENT__"].get<int>();
        }

        if (node_json.contains("instance")) {
            Variant instance_variant{Ref<Resource>()};
            if (!VariantSerializer::JsonToVariant(instance_variant, node_json["instance"], this)) {
                LOG_ERROR("Cannot load scene instance for node '{}'", node_data.name);
                return false;
            }

            bool success{false};
            Ref<Resource> instance_resource = instance_variant.convert<Ref<Resource>>(&success);
            node_data.instance = success ? dynamic_pointer_cast<PackedScene>(instance_resource) : Ref<PackedScene>();
            if (!node_data.instance.IsValid()) {
                LOG_ERROR("Scene instance for node '{}' is not a valid PackedScene.", node_data.name);
                return false;
            }
        }

        Type node_type = Type::get_by_name(node_data.type);
        if (!node_type.is_valid()) {
            LOG_ERROR("Unknown scene node type: {}", node_data.type);
            return false;
        }

        Variant new_obj = node_type.create();
        bool success{false};
        auto* node = new_obj.convert<Node*>(&success);
        if (!success || node == nullptr) {
            LOG_ERROR("Scene node type is not a Node: {}", node_data.type);
            return false;
        }

        if (node_json.contains("properties")) {
            const Json& properties_json = node_json["properties"];
            if (!properties_json.is_object()) {
                Object::Delete(node);
                LOG_ERROR("Scene node properties must be an object: {}", node_json.dump(4));
                return false;
            }

            for (const auto& [property_name, property_json] : properties_json.items()) {
                Variant property_value = node->Get(property_name);
                if (!property_value.is_valid()) {
                    LOG_ERROR("Unknown property '{}' on scene node type '{}'", property_name, node_data.type);
                    continue;
                }

                const Type property_type = node->GetPropertyType(property_name);
                if (property_json.is_null()) {
                    node_data.properties.push_back({property_name, property_value});
                    continue;
                }

                if (!VariantSerializer::JsonToVariant(property_value, property_json, this)) {
                    Object::Delete(node);
                    LOG_ERROR("Cannot load property '{}' for scene node type '{}'", property_name, node_data.type);
                    return false;
                }

                if (!property_value.convert(property_type)) {
                    Object::Delete(node);
                    LOG_ERROR("Cannot convert property '{}' to type '{}'", property_name, property_type.get_name().data());
                    return false;
                }

                node_data.properties.push_back({property_name, property_value});
            }
        }

        Object::Delete(node);
        state->AddNode(node_data);
    }

    return true;
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
    if (packed_scene_.IsValid()) {
        FindSceneResources(packed_scene_);
    } else {
        FindResources(resource, true);
    }

    if (packed_scene_.IsValid()) {
        // Add instances to external resources if saving a packed scene.
        for (std::size_t i = 0; i < packed_scene_->GetState()->GetNodeCount(); i++) {
            Ref<PackedScene> instance = packed_scene_->GetState()->GetNodeInstance(i);
            if (instance.IsValid() && !external_resources_.contains(instance)) {
                if (instance->GetPath().empty()) {
                    const SceneState::NodeData* node_data = packed_scene_->GetState()->GetNodeData(i);
                    const std::string node_name = node_data != nullptr ? node_data->name : std::string();
                    LOG_ERROR("Cannot save scene instance node '{}': referenced PackedScene has no resource path.",
                              node_name);
                    return false;
                }
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
        bool main = !packed_scene_.IsValid() && (saved_resource == saved_resources_.back());
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
                saved_resource->SetPathWithoutCache(path + "::" + saved_resource->GetUniqueId());
            }
            internal_resources_[saved_resource] = saved_resource->GetUniqueId();

            resource_data_json["__ID__"] = saved_resource->GetUniqueId();
            root["__SUB_RESOURCES__"].emplace_back(resource_data_json);
        }
    }

    if (packed_scene_.IsValid()) {
        root["__TYPE__"] = "PackedScene";
        root["__NODES__"] = SaveSceneNodes();
    }

    std::ofstream out_file(global_path);
    out_file << root.dump(4);
    out_file.close();
    return true;
}

void ResourceFormatSaverSceneInstance::FindSceneResources(const Ref<PackedScene>& packed_scene) {
    if (!packed_scene.IsValid() || !packed_scene->GetState().IsValid()) {
        return;
    }

    Ref<SceneState> state = packed_scene->GetState();
    for (std::size_t i = 0; i < state->GetNodeCount(); ++i) {
        const SceneState::NodeData* node_data = state->GetNodeData(i);
        if (node_data == nullptr) {
            continue;
        }

        if (node_data->instance.IsValid()) {
            Ref<Resource> instance_resource = node_data->instance;
            FindResources(Variant(instance_resource));
        }

        for (const auto& property : node_data->properties) {
            FindResources(property.value);
        }
    }
}

Json ResourceFormatSaverSceneInstance::SaveSceneNodes() {
    Json nodes_json = Json::array();
    if (!packed_scene_.IsValid() || !packed_scene_->GetState().IsValid()) {
        return nodes_json;
    }

    Ref<SceneState> state = packed_scene_->GetState();
    for (std::size_t i = 0; i < state->GetNodeCount(); ++i) {
        const SceneState::NodeData* node_data = state->GetNodeData(i);
        if (node_data == nullptr) {
            continue;
        }

        Json node_json;
        node_json["type"] = node_data->type;
        node_json["name"] = node_data->name;
        node_json["parent"] = node_data->parent;
        if (node_data->instance.IsValid()) {
            Ref<Resource> instance_resource = node_data->instance;
            node_json["instance"] = VariantSerializer::VariantToJson(Variant(instance_resource), this);
        }

        Json properties_json = Json::object();
        for (const auto& property : node_data->properties) {
            Json property_json = VariantSerializer::VariantToJson(property.value, this);
            if ((node_data->type == "MeshInstance3D" &&
                 (property.name == "mesh" || property.name == "material_override" || property.name == "material")) ||
                (node_data->type == "ArrayMesh" && property.name == "material")) {
                LOG_TRACE("Saving scene '{}' node '{}' type '{}' property '{}' as {}.",
                          local_path_,
                          node_data->name,
                          node_data->type,
                          property.name,
                          property_json.dump());
            }
            properties_json[property.name] = property_json;
        }
        node_json["properties"] = properties_json;

        nodes_json.push_back(node_json);
    }

    return nodes_json;
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
            if ((bool) (property_info.usage & PropertyUsageFlags::Storage)) {
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
