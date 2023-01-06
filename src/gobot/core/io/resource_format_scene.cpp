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

#include <QFile>

namespace gobot {

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

    local_path_ = ProjectSettings::GetInstance().LocalizePath(path);

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
    root["__VERSION__"] = 1.0;
    root["__META_TYPE__"] = packed_scene_.is_valid() ? "SCENE" : resource->GetClassName();

    root["__EXT_RESOURCES__"] = Json::array();
    for (const auto& [res, uuid]: external_resources_) {
        Json ext_res;
        ext_res["__MATA_TYPE__"] = res->GetClassName();
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
        } else {
            if (!root.contains("__SUB_RESOURCES__")) {
                root["__SUB_RESOURCES__"] = Json::object();
            }
            resource_data_json["__MATA_TYPE__"] = saved_resource->GetClassName();
            root["__SUB_RESOURCES__"][saved_resource->GetResourceUuid().toString().toStdString()] = resource_data_json;
        }
    }


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