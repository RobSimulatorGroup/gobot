/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * This file is created by Qiqi Wu, 22-11-20
*/

#include "gobot/core/ref_counted.hpp"
#include "gobot/core/io/resource.hpp"

namespace gobot {

enum class GOBOT_API ResourceSaverFlags {
    None = 0,
    ChangePath = 1,
};

class GOBOT_API ResourceFormatSaver : public RefCounted {
    GOBCLASS(ResourceFormatSaver, RefCounted);
public:
    virtual bool Save(const Ref<Resource> &resource, const String &path, ResourceSaverFlags flags = ResourceSaverFlags::None) = 0;

    virtual bool Recognize(const Ref<Resource> &p_resource) const = 0;

    virtual void GetRecognizedExtensions(const Ref<Resource> &resource, std::vector<String>* extensions) const = 0;

    virtual bool RecognizePath(const Ref<Resource> &resource, const String &path) const;
};

class GOBOT_API ResourceSaver {
public:

    using ResourceSavedCallback = std::function<void(Ref<Resource> resource, const String &path)>;

    static bool Save(const Ref<Resource> &resource, const String &target_path = "", ResourceSaverFlags flags = ResourceSaverFlags::None);

    static void GetRecognizedExtensions(const Ref<Resource> &resource, std::vector<String>* extensions);

    static void AddResourceFormatSaver(Ref<ResourceFormatSaver> format_saver, bool at_front = false);

    static void RemoveResourceFormatSaver(const Ref<ResourceFormatSaver>& format_saver);

    static void SetTimestampOnSave(bool timestamp) { s_timestamp_on_save = timestamp; }

    static bool GetTimestampOnSave() { return s_timestamp_on_save; }

    static void SetSaveCallback(ResourceSavedCallback callback);

private:
    static std::deque<Ref<ResourceFormatSaver>> s_savers;
    static bool s_timestamp_on_save;

    static ResourceSavedCallback resource_saved_callback;


};

}