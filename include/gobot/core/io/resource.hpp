/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * This file is created by Qiqi Wu, 22-11-20
*/

#pragma once

#include <mutex>
#include "gobot/core/ref_counted.hpp"
#include "gobot/core/object_id.hpp"
#include "gobot/core/rid.hpp"

namespace gobot {

class Node;

class GOBOT_EXPORT Resource: public RefCounted {
    GOBCLASS(Resource, RefCounted)
public:
    Resource();

    ~Resource();

    // Take over will delete old resource cache, further attempts to load an overridden resource by path will instead return this resource
    // If the take_over is false and same path resource cache already existed, this operation will fail.
    virtual void SetPath(const std::string &path, bool take_over = false);

    std::string GetPath() const;

    void SetName(const std::string &p_name);

    std::string GetName() const;

    // Generate 5 characters string made up with 0-9|a-z|A-Z
    static std::string GenerateResourceUniqueId();

    void SetUniqueId(const std::string &unique_id);

    std::string GetUniqueId() const;

    virtual void ReloadFromFile();

    static bool IsResourceFile(std::string_view path);

    // for resources that use various amount of properties, either via _validate_property or _get_property_list, this function needs to be implemented to correctly clear state
    virtual void ResetState();

    bool CopyFrom(const Ref<Resource> &resource);

    void RegisterOwner(Object *owner);

    void UnregisterOwner(Object *owner);

    FORCE_INLINE bool IsBuiltIn() const { return path_cache_.empty() || path_cache_.find("::") != std::string::npos; }

    Ref<Resource> CloneForLocalScene(Node* for_scene);

    // By default, sub-resources are shared between resource copies for efficiency.
    // If copy_subresource is false, a shallow copy is returned. Nested resources within subresources are not duplicated and are shared from the original resource.
    // This behavior can be overridden by the PropertyUsageFlags::NotSharedOnClone flag
    virtual Ref<Resource> Clone(bool copy_subresource = false) const;

    virtual RID GetRid() const; // some resources may offer conversion to RID

protected:
    void SetPathNotTakeOver(const std::string &path);

    void SetPathTakeOver(const std::string &path);

private:
    friend class ResourceCache;

    std::unordered_set<ObjectID> owners_;

    std::string name_;
    std::string path_cache_;
    Node *local_scene_ = nullptr;
    bool local_to_scene_{false};
    std::string unique_id_;

    GOBOT_REGISTRATION_FRIEND
};


class GOBOT_EXPORT ResourceCache {
public:
    static bool Has(const std::string &path);

    static Ref<Resource> GetRef(const std::string &path);

private:
    friend class Resource;
    friend class ResourceLoader;

    static std::recursive_mutex s_lock;

    // [path, resource]
    static std::unordered_map<std::string, Resource*> s_resources;

    static std::unordered_map<std::string, std::unordered_map<std::string, std::string>> s_resource_path_cache;

    static void Clear();


};

}