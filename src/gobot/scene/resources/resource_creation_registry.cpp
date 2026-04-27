/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/scene/resources/resource_creation_registry.hpp"

#include <algorithm>

#include "gobot/scene/resources/cylinder_shape_3d.hpp"
#include "gobot/scene/resources/material.hpp"
#include "gobot/scene/resources/mesh.hpp"
#include "gobot/scene/resources/primitive_mesh.hpp"
#include "gobot/scene/resources/shape_3d.hpp"

namespace gobot {

namespace {

std::vector<ResourceCreationEntry>& MutableResourceTypes() {
    static std::vector<ResourceCreationEntry> resource_types;
    return resource_types;
}

bool& BuiltInsRegistered() {
    static bool registered = false;
    return registered;
}

template <typename T>
Ref<Resource> CreateResourceRef() {
    return MakeRef<T>();
}

Type GetExpectedResourceType(const Type& property_type) {
    if (!property_type.is_wrapper() ||
        property_type.get_wrapper_holder_type() != WrapperHolderType::Ref) {
        return Type::get_by_name("");
    }

    Type wrapped_type = property_type.get_wrapped_type().get_raw_type();
    if (!wrapped_type.is_valid() || wrapped_type.is_pointer()) {
        wrapped_type = property_type.get_wrapped_type();
    }
    if (wrapped_type.is_pointer()) {
        wrapped_type = wrapped_type.get_raw_type();
    }
    return wrapped_type;
}

bool IsResourceTypeCompatible(const Type& resource_type, const Type& expected_type) {
    return resource_type == expected_type || resource_type.is_derived_from(expected_type);
}

Variant ResourceRefToPropertyVariant(const Ref<Resource>& resource, const Type& property_type) {
    if (!resource.IsValid()) {
        return {};
    }

    Type expected_type = GetExpectedResourceType(property_type);
    if (!expected_type.is_valid()) {
        return {};
    }

    if (expected_type == Type::get<Resource>()) {
        return resource;
    }
    if (expected_type == Type::get<Mesh>()) {
        return resource.DynamicPointerCast<Mesh>();
    }
    if (expected_type == Type::get<PrimitiveMesh>()) {
        return resource.DynamicPointerCast<PrimitiveMesh>();
    }
    if (expected_type == Type::get<BoxMesh>()) {
        return resource.DynamicPointerCast<BoxMesh>();
    }
    if (expected_type == Type::get<Material>()) {
        return resource.DynamicPointerCast<Material>();
    }
    if (expected_type == Type::get<PBRMaterial3D>()) {
        return resource.DynamicPointerCast<PBRMaterial3D>();
    }
    if (expected_type == Type::get<Shape3D>()) {
        return resource.DynamicPointerCast<Shape3D>();
    }
    if (expected_type == Type::get<CylinderShape3D>()) {
        return resource.DynamicPointerCast<CylinderShape3D>();
    }

    return {};
}

} // namespace

bool ResourceCreationRegistry::RegisterResourceType(ResourceCreationEntry entry) {
    if (entry.id.empty() || !entry.type.is_valid() || !entry.create_resource) {
        return false;
    }

    auto& resource_types = MutableResourceTypes();
    const auto existing = std::find_if(resource_types.begin(), resource_types.end(),
                                       [&entry](const ResourceCreationEntry& item) {
                                           return item.id == entry.id;
                                       });
    if (existing != resource_types.end()) {
        *existing = std::move(entry);
        return true;
    }

    resource_types.push_back(std::move(entry));
    return true;
}

const std::vector<ResourceCreationEntry>& ResourceCreationRegistry::GetResourceTypes() {
    EnsureBuiltInResourceTypesRegistered();
    return MutableResourceTypes();
}

const ResourceCreationEntry* ResourceCreationRegistry::FindResourceType(const std::string& id) {
    const auto& resource_types = GetResourceTypes();
    const auto iter = std::find_if(resource_types.begin(), resource_types.end(),
                                   [&id](const ResourceCreationEntry& item) {
                                       return item.id == id;
                                   });
    return iter == resource_types.end() ? nullptr : &(*iter);
}

Variant ResourceCreationRegistry::CreateResourceVariant(const std::string& id) {
    const auto* entry = FindResourceType(id);
    return entry == nullptr ? Variant() : Variant(entry->create_resource());
}

Variant ResourceCreationRegistry::CreateResourceVariant(const std::string& id, const Type& property_type) {
    const auto* entry = FindResourceType(id);
    if (entry == nullptr) {
        return {};
    }

    return ResourceRefToPropertyVariant(entry->create_resource(), property_type);
}

std::vector<const ResourceCreationEntry*> ResourceCreationRegistry::GetCreatableTypesForProperty(const Type& property_type) {
    Type expected_type = GetExpectedResourceType(property_type);
    if (!expected_type.is_valid()) {
        return {};
    }

    std::vector<const ResourceCreationEntry*> compatible_types;
    for (const auto& entry : GetResourceTypes()) {
        if (IsResourceTypeCompatible(entry.type, expected_type)) {
            compatible_types.push_back(&entry);
        }
    }

    std::sort(compatible_types.begin(), compatible_types.end(),
              [](const ResourceCreationEntry* lhs, const ResourceCreationEntry* rhs) {
                  return lhs->display_name < rhs->display_name;
              });

    return compatible_types;
}

void ResourceCreationRegistry::EnsureBuiltInResourceTypesRegistered() {
    if (BuiltInsRegistered()) {
        return;
    }

    BuiltInsRegistered() = true;

    RegisterResourceType({
        "PBRMaterial3D",
        "PBRMaterial3D",
        "Material",
        "Physically based 3D material.",
        Type::get<PBRMaterial3D>(),
        []() -> Ref<Resource> { return CreateResourceRef<PBRMaterial3D>(); }
    });

    RegisterResourceType({
        "BoxMesh",
        "BoxMesh",
        "PrimitiveMesh",
        "Box mesh resource.",
        Type::get<BoxMesh>(),
        []() -> Ref<Resource> { return CreateResourceRef<BoxMesh>(); }
    });

    RegisterResourceType({
        "CylinderShape3D",
        "CylinderShape3D",
        "Shape3D",
        "Cylinder collision shape resource.",
        Type::get<CylinderShape3D>(),
        []() -> Ref<Resource> { return CreateResourceRef<CylinderShape3D>(); }
    });
}

} // namespace gobot
