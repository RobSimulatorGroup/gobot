/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-20
*/

#include "gobot/core/io/resource.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/log.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/error_macros.hpp"


#include <random>

namespace gobot {

Resource::Resource() {

}

void Resource::RegisterOwner(Object *p_owner) {
    owners_.insert(p_owner->GetInstanceId());
}

void Resource::UnregisterOwner(Object *p_owner) {
    owners_.erase(p_owner->GetInstanceId());
}

Ref<Resource> Resource::CloneForLocalScene(Node* for_scene) {
    auto type = GetType();
    auto new_resource = type.create();
    ERR_FAIL_COND_V_MSG(!new_resource.is_valid() && new_resource.can_convert<Resource*>(),
                        nullptr, fmt::format("Failed to Create new Resource: {}", GetClassStringName()));

    Ref<Resource> r(new_resource.convert<Resource*>());

    r->local_scene_ = for_scene;

    for (const auto& prop : type.get_properties()) {
        if (prop.is_readonly()) {
            continue;
        }
        PropertyInfo property_info;
        auto meta_data = prop.get_metadata(PROPERTY_INFO_KEY);
        if (meta_data.is_valid()) {
            property_info = meta_data.get_value<PropertyInfo>();
        }
        USING_ENUM_BITWISE_OPERATORS;
        if (!(bool)(property_info.usage & PropertyUsageFlags::Storage)) {
            continue;
        }
        auto prop_value = Get(prop.get_name().data());
        if (prop.get_type().get_wrapper_holder_type() == WrapperHolderType::Ref) {
            if (!prop_value.can_convert<Ref<Resource>>()) {
                LOG_ERROR("prop:{} cannot convert to Ref<Resource>", prop.get_name().data());
            } else {
                auto re = prop_value.convert<Ref<Resource>>();
                if (re->local_to_scene_) {
                    Ref<Resource> prop_copy = re->CloneForLocalScene(for_scene);
                    re = prop_copy;
                }
            }
        }
        r->Set(prop.get_name().data(), prop_value);
    }

    return r;
}

Ref<Resource> Resource::Clone(bool copy_subresource) const {
    auto type = GetType();
    auto new_resource = type.create();
    ERR_FAIL_COND_V_MSG(!new_resource.is_valid() && new_resource.can_convert<Resource*>(),
                        nullptr, fmt::format("Failed to Create new Resource: {}", GetClassStringName()));


    Ref<Resource> r(new_resource.convert<Resource*>());

    for (const auto& prop : type.get_properties()) {
        if (prop.is_readonly()) {
            continue;
        }
        PropertyInfo property_info;
        auto meta_data = prop.get_metadata(PROPERTY_INFO_KEY);
        if (meta_data.is_valid()) {
            property_info = meta_data.get_value<PropertyInfo>();
        }
        USING_ENUM_BITWISE_OPERATORS;
        if (!(bool)(property_info.usage & PropertyUsageFlags::Storage)) {
            continue;
        }
        auto prop_name = prop.get_name().data();
        Variant p = Get(prop_name);
        if (p.get_type().get_wrapper_holder_type() == WrapperHolderType::Ref &&
                  (copy_subresource || (bool)(property_info.usage & PropertyUsageFlags::NotSharedOnClone))) {
            if (!p.can_convert<Ref<Resource>>()) {
                LOG_ERROR("prop:{} cannot convert to Ref<Resource>", prop.get_name().data());
            } else {
                auto sr = p.convert<Ref<Resource>>();
                r->Set(prop_name, sr->Clone(copy_subresource));
            }
        } else {
            r->Set(prop_name, p);
        }
    }

    return r;
}

Resource::~Resource() {
    if (!path_cache_.isEmpty()) {
        ResourceCache::s_lock.lock();
        ResourceCache::s_resources.erase(path_cache_);
        ResourceCache::s_lock.unlock();
    }
    if (!owners_.empty()) {
        LOG_WARN("Resource is still owned.");
    }
}

void Resource::SetPathNotTakeOver(const String &path) {
    SetPath(path, false);
}

void Resource::SetPathTakeOver(const String &path) {
    SetPath(path, true);
}

void Resource::SetPath(const String &path, bool take_over) {
    if (path_cache_ == path) {
        return;
    }

    if (path.isEmpty()) {
        take_over = false; // Can't take over an empty path
    }

    ResourceCache::s_lock.lock();

    if (!path_cache_.isEmpty()) {
        ResourceCache::s_resources.erase(path_cache_);
    }

    path_cache_.clear();

    Ref<Resource> existing = ResourceCache::GetRef(path);

    if (existing.UseCount()) {
        if (take_over) {
            existing->path_cache_ = String();
            ResourceCache::s_resources.erase(path);
        } else {
            ResourceCache::s_lock.unlock();
            LOG_ERROR("Another resource is loaded from path {} (possible cyclic resource inclusion).", path);
            return;
        }
    }

    path_cache_ = path;

    if (!path_cache_.isEmpty()) {
        ResourceCache::s_resources[path_cache_] = this;
    }
    ResourceCache::s_lock.unlock();

}

String Resource::GetPath() const {
    return path_cache_;
}

void Resource::SetName(const String &name) {
    name_ = name;
    Q_EMIT resourceChanged();
}

String Resource::GetName() const {
    return name_;
}

void Resource::SetUniqueId(const String &unique_id) {
    unique_id_ = unique_id;
}

String Resource::GetUniqueId() const {
    return unique_id_;
}

String Resource::GenerateResourceUniqueId() {
    int length = 5;

    static auto& chrs = "0123456789"
                        "abcdefghijklmnopqrstuvwxyz"
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    thread_local static std::mt19937 rg{std::random_device{}()};
    thread_local static std::uniform_int_distribution<> pick(0, sizeof(chrs) - 2);

    String s;
    s.reserve(length);

    while(length--)
        s += chrs[pick(rg)];
    return s;
}


bool Resource::IsResourceFile(const String& path) {
    // "::" means property name
    return path.startsWith("res://") && !path.contains("::");
}

void Resource::ReloadFromFile() {
    auto path = GetPath();

    ERR_FAIL_COND(!IsResourceFile(path));

    Ref<Resource> resource = ResourceLoader::Load(path, GetClassStringName().data(), ResourceFormatLoader::CacheMode::Ignore);

    ERR_FAIL_COND(!resource.IsValid());

    CopyFrom(resource);
}

void Resource::ResetState() {

}

bool Resource::CopyFrom(const Ref<Resource> &resource) {
    ERR_FAIL_COND_V_MSG(!resource.IsValid(), false, "Input resource is invalid");
    ERR_FAIL_COND_V_MSG(GetClassStringName() != resource->GetClassStringName(), false,
                        fmt::format("Input resource's type:{} is not same as this type: {}", resource->GetClassStringName(),
                        GetClassStringName()));

    ResetState(); //may want to reset state

    for (const auto& prop : resource->get_type().get_properties()) {
        if (prop.is_readonly()) {
            continue;
        }
        PropertyInfo property_info;
        auto meta_data = prop.get_metadata(PROPERTY_INFO_KEY);
        if (meta_data.is_valid()) {
            property_info = meta_data.get_value<PropertyInfo>();
        }
        USING_ENUM_BITWISE_OPERATORS;
        if (!(bool)(property_info.usage & PropertyUsageFlags::Storage)) {
            continue;
        }
        String prop_name = prop.get_name().data();
        if (prop_name == "resource_path") {
            continue; //do not change path
        }
        Set(prop_name, resource->Get(prop_name));
    }
    return true;
}

RID Resource::GetRid() const {
    return {};
}


std::unordered_map<String, Resource*> ResourceCache::s_resources;
std::recursive_mutex ResourceCache::s_lock;

bool ResourceCache::Has(const String &path) {
    s_lock.lock();

    auto it = s_resources.find(path);

    if (it != s_resources.end() && it->second->GetReferenceCount() == 0) {
        // This resource is in the process of being deleted, ignore its existence.
        it->second->path_cache_ = String();
        it->second = nullptr;
        s_resources.erase(path);
    }

    s_lock.unlock();

    if (it == s_resources.end()) {
        return false;
    }

    return true;
}

Ref<Resource> ResourceCache::GetRef(const String &path) {
    Ref<Resource> ref;
    s_lock.lock();

    auto it = s_resources.find(path);

    if (it != s_resources.end()) {
        ref = Ref<Resource>(it->second);
    }

    if (it != s_resources.end() && it->second->GetReferenceCount() == 0) {
        // This resource is in the process of being deleted, ignore its existence
        it->second->path_cache_ = String();
        it->second = nullptr;
        s_resources.erase(path);
    }

    s_lock.unlock();

    return ref;
}

void ResourceCache::Clear() {
    if (!s_resources.empty()) {
        LOG_ERROR("Resources still in use at exit");
#ifdef NDEBUG
#else
        for (const auto& [path, resource]: s_resources) {
            LOG_TRACE("Resource:{} with path:{} is still in use", path, resource->GetClassStringName());
        }
#endif
    }

    s_resources.clear();
}


}

GOBOT_REGISTRATION {
    Class_<Resource>("Resource")
        .constructor()(CtorAsRawPtr)
        .property("resource_name", &Resource::GetName, &Resource::SetName)
        .property("resource_path", &Resource::GetPath, &Resource::SetPathNotTakeOver);

    gobot::Type::register_wrapper_converter_for_base_classes<Ref<Resource>, Ref<RefCounted>>();

};
