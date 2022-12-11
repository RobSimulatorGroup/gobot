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

namespace gobot {

class Node;

class Resource: public RefCounted {
    Q_OBJECT
    GOBCLASS(Resource, RefCounted)
public:
    Resource();

    ~Resource();

    virtual void SetPath(const String &path, bool take_over = false);

    String GetPath() const;

    void SetName(const String &p_name);

    String GetName() const;

    void SetResourceUuid(const Uuid &uuid);

    Uuid GetResourceUuid() const;

    Uuid GenerateUuid();

    virtual void ReloadFromFile();

    static bool IsResourceFile(const String& path);

    // for resources that use variable amount of properties, either via _validate_property or _get_property_list, this function needs to be implemented to correctly clear state
    virtual void ResetState();

    bool CopyFrom(const Ref<Resource> &resource);

    void RegisterOwner(Object *owner);

    void UnregisterOwner(Object *owner);

    Ref<Resource> DuplicateForLocalScene(Node* for_scene);

    // If subresources is false, a shallow copy is returned.
    // Nested resources within subresources are not duplicated and are shared from the original resource.
    // This behavior can be overridden by the PropertyUsageFlags::NotSharedOnClone flag
    virtual Ref<Resource> Clone(bool subresources = false) const;

protected:
    void SetPath(const String &path);

Q_SIGNALS:
    void resourceChanged();

private:
    friend class ResourceCache;

    std::unordered_set<ObjectID> owners_;

    String name_;
    String path_cache_;
    Node *local_scene_ = nullptr;
    bool local_to_scene_{false};
    Uuid uuid_{};

    GOBOT_REGISTRATION_FRIEND
};


class ResourceCache {
public:
    static bool Has(const String &path);

    static Ref<Resource> GetRef(const String &path);

private:
    friend class Resource;
    friend class ResourceLoader;

    static std::mutex s_lock;
    static std::unordered_map<String, Resource*> s_resources;

    static void Clear();


};

}