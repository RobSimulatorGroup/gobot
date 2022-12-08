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

namespace gobot {

Resource::Resource() {

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

void Resource::SetPath(const String &path) {
    SetPath(path, false);
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

    if (existing.use_count()) {
        if (take_over) {
            existing->path_cache_ = String();
            ResourceCache::s_resources.erase(path);
        } else {
            ResourceCache::s_lock.unlock();
            LOG_ERROR("Another resource is loaded from path {} (possible cyclic resource inclusion).", path);
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

void Resource::SetResourceUuid(const Uuid &uuid) {
    uuid_ = uuid;
}

Uuid Resource::GetResourceUuid() const {
    return uuid_;
}

Uuid Resource::GenerateUuid() {
    uuid_ = Uuid::createUuid();
    return uuid_;
}


bool Resource::IsResourceFile(const String& path) {
    // "::" means property name
    return path.startsWith("res://") && !path.contains("::");
}

void Resource::ReloadFromFile() {
    auto path = GetPath();
    if (!IsResourceFile(path)) {
        return;
    }

    Ref<Resource> resource = ResourceLoader::Load(path, GetClassName().data(), ResourceFormatLoader::CacheMode::Ignore);

    if (!resource.is_valid()) {
        return;
    }

    CopyFrom(resource);
}

void Resource::ResetState() {

}

bool Resource::CopyFrom(const Ref<Resource> &resource) {
    if (!resource.is_valid()) {
        LOG_ERROR("input resource is invalid");
        return false;
    }

    if (GetClassName() != resource->GetClassName()) {
        LOG_ERROR("input resource's type:{} is not same as this type: {}", resource->GetClassName(), GetClassName());
        return false;
    }

    ResetState(); //may want to reset state

    for (const auto& prop : resource->get_type().get_properties()) {
        auto property_info = prop.get_metadata("").get_value<PropertyInfo>();
        USING_ENUM_BITWISE_OPERATORS;
        if (!(bool)(property_info.usage & PropertyUsageFlags::Storage)) {
            continue;
        }
        if (property_info.name == "resource_path") {
            continue; //do not change path
        }
        Set(property_info.name, resource->Get(property_info.name));
    }
    return true;
}


std::unordered_map<String, Resource*> ResourceCache::s_resources;
std::mutex ResourceCache::s_lock;

bool ResourceCache::Has(const String &path) {
    s_lock.lock();

    auto it = s_resources.find(path);

    if (it != s_resources.end() && it->second->use_count() == 0) {
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

    if (it == s_resources.end() && it->second->use_count() == 0) {
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
            LOG_TRACE("Resource:{} with path:{} is still in use", path, resource->GetClassName());
        }
#endif
    }

    s_resources.clear();
}


}

GOBOT_REGISTRATION {
    Class_<Resource>("Resource")
        .constructor()(CtorAsRawPtr)
        .property("name", &Resource::GetName, &Resource::SetName)
        .property("resource_path", &Resource::GetPath, overload_cast<const String &>(&Resource::SetPath));

};
