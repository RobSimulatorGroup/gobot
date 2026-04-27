/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "gobot/core/io/resource.hpp"

namespace gobot {

struct ResourceCreationEntry {
    std::string id;
    std::string display_name;
    std::string parent_id;
    std::string description;
    Type type;
    std::function<Ref<Resource>()> create_resource;
};

class GOBOT_EXPORT ResourceCreationRegistry {
public:
    static bool RegisterResourceType(ResourceCreationEntry entry);

    static const std::vector<ResourceCreationEntry>& GetResourceTypes();

    static const ResourceCreationEntry* FindResourceType(const std::string& id);

    static Variant CreateResourceVariant(const std::string& id);

    static Variant CreateResourceVariant(const std::string& id, const Type& property_type);

    static std::vector<const ResourceCreationEntry*> GetCreatableTypesForProperty(const Type& property_type);

private:
    static void EnsureBuiltInResourceTypesRegistered();
};

}
