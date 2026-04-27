/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/editor/property_inspector/editor_property_resource.hpp"

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "gobot/core/io/resource.hpp"
#include "gobot/editor/imgui/type_icons.hpp"
#include "gobot/editor/property_inspector/editor_inspector.hpp"
#include "gobot/scene/resources/resource_creation_registry.hpp"
#include "imgui.h"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"

namespace gobot {

namespace {

struct ResourceInspectorContext {
    int depth = 0;
    std::unordered_set<const Resource*> visited;
};

ResourceInspectorContext& GetResourceInspectorContext() {
    thread_local ResourceInspectorContext context;
    return context;
}

Ref<Resource> GetResourceRef(const Variant& variant) {
    bool success = false;
    Ref<Resource> resource = variant.convert<Ref<Resource>>(&success);
    return success ? resource : Ref<Resource>();
}

bool AssignResourceVariant(PropertyDataModel* property_data_model, Variant resource_variant) {
    if (!resource_variant.is_valid()) {
        return false;
    }

    if (!resource_variant.convert(property_data_model->GetValueType())) {
        return false;
    }

    return property_data_model->SetValue(resource_variant);
}

std::vector<Type> BuildInheritanceChain(Type type) {
    std::vector<Type> chain;
    for (const auto& base : type.get_base_classes()) {
        chain.emplace_back(base);
    }
    chain.emplace_back(type);
    std::reverse(chain.begin(), chain.end());
    return chain;
}

std::vector<Property> GetDeclaredEditableProperties(Type type) {
    std::vector<Property> properties;
    for (const auto& property : type.get_properties(rttr::filter_item::public_access |
                                                    rttr::filter_item::declared_only |
                                                    rttr::filter_item::static_item |
                                                    rttr::filter_item::instance_item)) {
        properties.emplace_back(property);
    }
    return properties;
}

void DrawResourceInspector(Resource* resource,
                           ResourceInspectorContext& context) {
    if (resource == nullptr) {
        return;
    }

    if (context.depth > 4) {
        ImGui::TextDisabled("Max nested resource depth reached.");
        return;
    }

    if (context.visited.contains(resource)) {
        ImGui::TextDisabled("Recursive resource reference.");
        return;
    }

    context.visited.insert(resource);
    context.depth++;

    Variant resource_variant(resource);
    VariantCache cache(resource_variant);
    const auto inheritance_chain = BuildInheritanceChain(cache.type);

    ImGui::PushID(resource);
    for (const auto& type : inheritance_chain) {
        auto properties = GetDeclaredEditableProperties(type);
        if (properties.empty()) {
            continue;
        }

        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_DefaultOpen |
                                   ImGuiTreeNodeFlags_Framed |
                                   ImGuiTreeNodeFlags_SpanAvailWidth |
                                   ImGuiTreeNodeFlags_FramePadding;
        const bool open = ImGui::TreeNodeEx(type.get_name().data(),
                                            flags,
                                            "%s %s",
                                            GetTypeIcon(type),
                                            type.get_name().data());
        if (!open) {
            continue;
        }

        for (const auto& property : properties) {
            auto editor = EditorInspectorDefaultPlugin::GetEditorForProperty(
                    std::make_unique<PropertyDataModel>(cache, property));
            if (editor == nullptr) {
                continue;
            }

            editor->OnImGui();
            Object::Delete(editor);
        }

        ImGui::TreePop();
    }
    ImGui::PopID();

    context.depth--;
    context.visited.erase(resource);
}

} // namespace

void EditorPropertyResource::OnImGuiContent() {
    Ref<Resource> resource = GetResourceRef(property_data_model_->GetValue());
    if (!resource.IsValid()) {
        ImGui::TextDisabled("<empty>");
    } else {
        ImGui::TextUnformatted(GetTypeIcon(resource->GetType()));
        ImGui::SameLine();
        ImGui::TextUnformatted(resource->GetType().get_name().data());

        const std::string path = resource->GetPath();
        if (!path.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("%s", path.c_str());
        }
    }

    ImGui::SameLine();
    if (ImGui::SmallButton(ICON_MDI_PLUS)) {
        ImGui::OpenPopup("CreateResource");
    }

    if (ImGui::BeginPopup("CreateResource")) {
        const auto creatable_types = ResourceCreationRegistry::GetCreatableTypesForProperty(
                property_data_model_->GetValueType());
        if (creatable_types.empty()) {
            ImGui::TextDisabled("No compatible resource types.");
        }

        for (const auto* entry : creatable_types) {
            if (ImGui::MenuItem(entry->display_name.c_str())) {
                AssignResourceVariant(property_data_model_,
                                      ResourceCreationRegistry::CreateResourceVariant(
                                              entry->id,
                                              property_data_model_->GetValueType()));
            }
        }
        ImGui::EndPopup();
    }
}

void EditorPropertyResource::End() {
    Ref<Resource> resource = GetResourceRef(property_data_model_->GetValue());

    EditorBuiltInProperty::End();

    if (!resource.IsValid()) {
        return;
    }

    ImGui::PushID(this);
    const bool open = ImGui::TreeNodeEx("ResourceProperties",
                                        ImGuiTreeNodeFlags_DefaultOpen |
                                        ImGuiTreeNodeFlags_SpanAvailWidth,
                                        "%s properties",
                                        resource->GetType().get_name().data());
    if (open) {
        DrawResourceInspector(resource.Get(), GetResourceInspectorContext());
        ImGui::TreePop();
    }
    ImGui::PopID();
}

} // namespace gobot
