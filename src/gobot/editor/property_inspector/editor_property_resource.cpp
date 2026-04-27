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

ImU32 GetResourceGroupBgColor(int depth) {
    const ImVec4 base = ImGui::GetStyleColorVec4(ImGuiCol_FrameBg);
    const float tint = depth % 2 == 0 ? 0.10f : 0.15f;
    return ImGui::ColorConvertFloat4ToU32({
            std::min(base.x + tint, 1.0f),
            std::min(base.y + tint * 0.75f, 1.0f),
            std::min(base.z + tint * 0.35f, 1.0f),
            0.92f
    });
}

void DrawResourceMenu(PropertyDataModel* property_data_model) {
    if (!ImGui::BeginPopup("ResourceMenu")) {
        return;
    }

    const auto creatable_types = ResourceCreationRegistry::GetCreatableTypesForProperty(
            property_data_model->GetValueType());
    if (creatable_types.empty()) {
        ImGui::TextDisabled("No compatible resource types.");
    } else {
        ImGui::TextDisabled("New");
        ImGui::Separator();
    }

    for (const auto* entry : creatable_types) {
        if (ImGui::MenuItem(entry->display_name.c_str())) {
            AssignResourceVariant(property_data_model,
                                  ResourceCreationRegistry::CreateResourceVariant(
                                          entry->id,
                                          property_data_model->GetValueType()));
        }
    }

    ImGui::EndPopup();
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
    const float dropdown_width = ImGui::GetFrameHeight();
    const float slot_width = std::max(1.0f,
                                      ImGui::GetContentRegionAvail().x -
                                      dropdown_width -
                                      ImGui::GetStyle().ItemSpacing.x);

    std::string slot_label;
    if (!resource.IsValid()) {
        slot_label = "<empty>";
    } else {
        slot_label = std::string(GetTypeIcon(resource->GetType())) + "  " +
                     resource->GetType().get_name().data();

        const std::string path = resource->GetPath();
        if (!path.empty()) {
            slot_label += "  ";
            slot_label += path;
        }
    }

    ImGui::Selectable(slot_label.c_str(), false, ImGuiSelectableFlags_None,
                      {slot_width, ImGui::GetFrameHeight()});
    ImGui::SameLine();
    if (ImGui::Button(ICON_MDI_MENU_DOWN, {dropdown_width, ImGui::GetFrameHeight()})) {
        ImGui::OpenPopup("ResourceMenu");
    }
    DrawResourceMenu(property_data_model_);
}

void EditorPropertyResource::End() {
    Ref<Resource> resource = GetResourceRef(property_data_model_->GetValue());

    EditorBuiltInProperty::End();

    if (!resource.IsValid()) {
        return;
    }

    ImGui::PushID(this);
    auto& context = GetResourceInspectorContext();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImVec2 bg_min = ImGui::GetCursorScreenPos();
    const float bg_max_x = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;

    draw_list->ChannelsSplit(2);
    draw_list->ChannelsSetCurrent(1);
    ImGui::BeginGroup();

    const bool open = ImGui::TreeNodeEx("ResourceProperties",
                                        ImGuiTreeNodeFlags_DefaultOpen |
                                        ImGuiTreeNodeFlags_SpanAvailWidth,
                                        "%s properties",
                                        resource->GetType().get_name().data());
    if (open) {
        ImGui::Indent(ImGui::GetStyle().IndentSpacing);
        DrawResourceInspector(resource.Get(), context);
        ImGui::Unindent(ImGui::GetStyle().IndentSpacing);
        ImGui::TreePop();
    }

    ImGui::EndGroup();
    const ImVec2 bg_max = {bg_max_x,
                           ImGui::GetItemRectMax().y + ImGui::GetStyle().ItemSpacing.y * 0.5f};
    draw_list->ChannelsSetCurrent(0);
    draw_list->AddRectFilled(bg_min, bg_max, GetResourceGroupBgColor(context.depth), 4.0f);
    draw_list->AddRect(bg_min, bg_max, ImGui::GetColorU32(ImGuiCol_Border), 4.0f);
    draw_list->ChannelsMerge();
    ImGui::PopID();
}

} // namespace gobot
