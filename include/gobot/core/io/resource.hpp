/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * This file is created by Qiqi Wu, 22-11-20
*/

#include "gobot/core/ref_counted.hpp"
#include <mutex>

namespace gobot {

class Resource: public RefCounted {
    Q_OBJECT
    GOBCLASS(Resource, RefCounted)
public:
    Resource();

    virtual void SetPath(const String &path);

    String GetPath() const;

    void SetName(const String &p_name);

    String GetName() const;

    void SetResourceUuid(const QUuid &uuid);

    Uuid GetResourceUuid() const;

    Uuid GenerateUuid();


Q_SIGNALS:
    void resourceChanged();

private:
    friend class ResourceCache;

    String name_;
    String path_cache_;
    bool local_to_scene_{false};
    Uuid uuid_{};

};


class ResourceCache {
public:
    static bool Has(const String &path);

    static Ref<Resource> GetRef(const String &path);

private:
    friend class Resource;
    friend class ResourceLoader; //need the lock
    static std::mutex s_lock;
    static std::unordered_map<String, Resource*> s_resources;

    static void Clear();

};

}