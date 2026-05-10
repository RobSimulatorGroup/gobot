/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#include "gobot/editor/property_inspector/editor_property_resource.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/io/python_script.hpp"
#include "gobot/core/io/resource.hpp"
#include "gobot/core/io/resource_loader.hpp"
#include "gobot/core/string_utils.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/editor/imgui/type_icons.hpp"
#include "gobot/editor/property_inspector/editor_inspector.hpp"
#include "gobot/log.hpp"
#include "gobot/scene/node.hpp"
#include "gobot/scene/resources/array_mesh.hpp"
#include "gobot/scene/resources/material.hpp"
#include "gobot/scene/resources/mesh.hpp"
#include "gobot/scene/resources/resource_creation_registry.hpp"
#include "imgui.h"
#include "imgui_internal.h"
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

std::string PythonScriptTemplate() {
    return "import gobot\n\n"
           "\n"
           "class Script(gobot.NodeScript):\n"
           "    def _ready(self):\n"
           "        pass\n"
           "\n"
           "    def _process(self, delta: float):\n"
           "        pass\n"
           "\n"
           "    def _physics_process(self, delta: float):\n"
           "        pass\n";
}

std::string SanitizeScriptBaseName(std::string value) {
    if (value.empty()) {
        value = "node";
    }
    std::ranges::transform(value, value.begin(), [](unsigned char character) {
        if (std::isalnum(character)) {
            return static_cast<char>(std::tolower(character));
        }
        return '_';
    });
    while (value.find("__") != std::string::npos) {
        value = ReplaceAll(value, "__", "_");
    }
    while (!value.empty() && value.front() == '_') {
        value.erase(value.begin());
    }
    while (!value.empty() && value.back() == '_') {
        value.pop_back();
    }
    return value.empty() ? "node" : value;
}

std::string UniqueScriptPathForNode(const Node& node) {
    const std::string base_name = SanitizeScriptBaseName(node.GetName()) + "_script";
    std::string candidate = "res://scripts/" + base_name + ".py";
    for (int index = 1; ResourceLoader::Exists(candidate, "PythonScript"); ++index) {
        candidate = "res://scripts/" + base_name + "_" + std::to_string(index) + ".py";
    }
    return candidate;
}

bool CreateScriptFile(const std::string& local_path) {
    const std::string global_path = ProjectSettings::GetInstance()->GlobalizePath(local_path);
    std::error_code error;
    std::filesystem::create_directories(std::filesystem::path(global_path).parent_path(), error);
    if (error) {
        LOG_ERROR("Failed to create Python script directory '{}': {}",
                  std::filesystem::path(global_path).parent_path().string(),
                  error.message());
        return false;
    }

    std::ofstream output(global_path, std::ios::out | std::ios::trunc);
    if (!output.is_open()) {
        LOG_ERROR("Failed to create Python script '{}'.", local_path);
        return false;
    }
    output << PythonScriptTemplate();
    return true;
}

bool AttachPythonScript(PropertyDataModel* property_data_model) {
    auto* node = Object::PointerCastTo<Node>(property_data_model->GetVariantCache().object);
    if (node == nullptr) {
        return false;
    }

    const std::string local_path = UniqueScriptPathForNode(*node);
    if (!CreateScriptFile(local_path)) {
        return false;
    }

    Ref<Resource> resource = ResourceLoader::Load(local_path, "PythonScript", ResourceFormatLoader::CacheMode::Replace);
    Ref<PythonScript> script = dynamic_pointer_cast<PythonScript>(resource);
    if (!script.IsValid()) {
        LOG_ERROR("Failed to load Python script '{}'.", local_path);
        return false;
    }

    if (!AssignResourceVariant(property_data_model, Variant(script))) {
        return false;
    }

    if (auto* editor = Editor::GetInstanceOrNull()) {
        editor->RefreshResourcePanel();
        editor->OpenPythonScriptFromPath(local_path);
    }
    return true;
}

bool IsNodeScriptProperty(PropertyDataModel* property_data_model) {
    return property_data_model->GetPropertyName() == "script" &&
           (property_data_model->GetHolderType() == Type::get<Node>() ||
            property_data_model->GetHolderType().is_derived_from(Type::get<Node>()));
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

std::vector<Property> GetEditableResourceProperties(Type type) {
    std::vector<Property> properties;
    for (const Type& inherited_type : BuildInheritanceChain(type)) {
        auto declared_properties = GetDeclaredEditableProperties(inherited_type);
        properties.insert(properties.end(), declared_properties.begin(), declared_properties.end());
    }
    return properties;
}

bool ShouldShowNestedResourceProperty(const Resource* resource, const Property& property) {
    if (resource == nullptr) {
        return false;
    }

    PropertyInfo property_info;
    const Variant property_metadata = property.get_metadata(PROPERTY_INFO_KEY);
    if (property_metadata.is_valid()) {
        property_info = property_metadata.get_value<PropertyInfo>();
    }

    USING_ENUM_BITWISE_OPERATORS;
    if (!static_cast<bool>(property_info.usage & PropertyUsageFlags::Editor)) {
        return false;
    }

    // Mesh material is an asset-level default. MeshInstance3D.material is the
    // scene instance override users edit from the node Inspector.
    if (Object::PointerCastTo<Mesh>(const_cast<Resource*>(resource)) != nullptr &&
        property.get_name() == "material") {
        return false;
    }

    return true;
}

ImVec4 WithAlpha(ImVec4 color, float alpha) {
    color.w = alpha;
    return color;
}

ImVec4 ScaleColor(ImVec4 color, float scale) {
    return {
            std::clamp(color.x * scale, 0.0f, 1.0f),
            std::clamp(color.y * scale, 0.0f, 1.0f),
            std::clamp(color.z * scale, 0.0f, 1.0f),
            color.w
    };
}

ImVec4 GetResourceTypeColor(Type type) {
    static constexpr std::array<ImVec4, 8> palette = {
            ImVec4{0.27f, 0.42f, 0.67f, 1.0f},
            ImVec4{0.55f, 0.35f, 0.72f, 1.0f},
            ImVec4{0.28f, 0.56f, 0.45f, 1.0f},
            ImVec4{0.68f, 0.43f, 0.24f, 1.0f},
            ImVec4{0.65f, 0.32f, 0.38f, 1.0f},
            ImVec4{0.42f, 0.50f, 0.25f, 1.0f},
            ImVec4{0.24f, 0.52f, 0.62f, 1.0f},
            ImVec4{0.48f, 0.42f, 0.70f, 1.0f},
    };

    const std::string type_name = type.get_name().data();
    const std::size_t index = std::hash<std::string>{}(type_name) % palette.size();
    return palette[index];
}

void PushResourceHeaderStyle(Type type) {
    const ImVec4 color = GetResourceTypeColor(type);
    ImGui::PushStyleColor(ImGuiCol_Header, WithAlpha(color, 0.78f));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, WithAlpha(ScaleColor(color, 1.12f), 0.88f));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, WithAlpha(ScaleColor(color, 1.22f), 0.96f));
}

void PopResourceHeaderStyle() {
    ImGui::PopStyleColor(3);
}

void DrawResourceBodyAccent(const ImVec2& min, const ImVec2& max, Type type) {
    if (max.y <= min.y) {
        return;
    }

    const ImVec4 color = GetResourceTypeColor(type);
    ImGui::GetWindowDrawList()->AddRectFilled(
            min,
            {min.x + 3.0f, max.y},
            ImGui::ColorConvertFloat4ToU32(WithAlpha(color, 0.95f)),
            1.0f);
}

void DrawResourceMenu(PropertyDataModel* property_data_model) {
    if (!ImGui::BeginPopup("ResourceMenu")) {
        return;
    }

    if (IsNodeScriptProperty(property_data_model)) {
        Ref<Resource> current = GetResourceRef(property_data_model->GetValue());
        if (!current.IsValid()) {
            if (ImGui::MenuItem(ICON_MDI_LANGUAGE_PYTHON " Attach Script")) {
                AttachPythonScript(property_data_model);
            }
        } else {
            ImGui::TextDisabled("Script is attached from SceneTree.");
        }
        ImGui::EndPopup();
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

void DrawClippedResourceSlot(const char* text, const ImVec2& size) {
    ImGui::Selectable("##ResourceSlot", false, ImGuiSelectableFlags_None, size);

    const ImVec2 min = ImGui::GetItemRectMin();
    const ImVec2 max = ImGui::GetItemRectMax();
    const ImVec2 padding = ImGui::GetStyle().FramePadding;
    ImGui::RenderTextClipped({min.x + padding.x, min.y},
                             {max.x - padding.x, max.y},
                             text,
                             nullptr,
                             nullptr,
                             {0.0f, 0.5f});
}

float GetAvailableResourceSlotWidth() {
    float available_width = ImGui::GetContentRegionAvail().x;
    if (ImGuiTable* table = ImGui::GetCurrentTable()) {
        const int column_index = ImGui::TableGetColumnIndex();
        if (column_index >= 0) {
            const ImRect cell_rect = ImGui::TableGetCellBgRect(table, column_index);
            const float cell_available_width = cell_rect.Max.x -
                                               ImGui::GetCursorScreenPos().x -
                                               ImGui::GetStyle().CellPadding.x;
            available_width = std::min(available_width, cell_available_width);
        }
    }

    return std::max(1.0f, available_width);
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
    const auto properties = GetEditableResourceProperties(cache.type);

    ImGui::PushID(resource);
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_DefaultOpen |
                               ImGuiTreeNodeFlags_Framed |
                               ImGuiTreeNodeFlags_SpanAvailWidth |
                               ImGuiTreeNodeFlags_FramePadding;
    PushResourceHeaderStyle(cache.type);
    const bool open = ImGui::TreeNodeEx(cache.type.get_name().data(),
                                        flags,
                                        "%s %s",
                                        GetTypeIcon(cache.type),
                                        cache.type.get_name().data());
    PopResourceHeaderStyle();
    if (open) {
        if (properties.empty()) {
            ImGui::TextDisabled("No editable properties.");
        } else {
            const ImVec2 body_min = ImGui::GetCursorScreenPos();
            ImGui::Indent(ImGui::GetStyle().IndentSpacing * 0.75f);
            for (const auto& property : properties) {
                if (!ShouldShowNestedResourceProperty(resource, property)) {
                    continue;
                }

                auto editor = EditorInspectorDefaultPlugin::GetEditorForProperty(
                        std::make_unique<PropertyDataModel>(cache, property));
                if (editor == nullptr) {
                    continue;
                }

                editor->OnImGui();
                Object::Delete(editor);
            }
            ImGui::Unindent(ImGui::GetStyle().IndentSpacing * 0.75f);
            const ImVec2 body_max = {
                    ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x,
                    ImGui::GetItemRectMax().y + ImGui::GetStyle().ItemSpacing.y * 0.25f
            };
            DrawResourceBodyAccent(body_min, body_max, cache.type);
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
    const float available_width = GetAvailableResourceSlotWidth();
    const float slot_width = std::max(1.0f,
                                      available_width -
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

    DrawClippedResourceSlot(slot_label.c_str(), {slot_width, ImGui::GetFrameHeight()});
    if (resource.IsValid() && ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_NoSharedDelay)) {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(slot_label.c_str());
        ImGui::EndTooltip();
    }
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

    DrawResourceInspector(resource.Get(), context);

    ImGui::PopID();
}

} // namespace gobot
