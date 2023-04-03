/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-28
*/

#include "gobot/editor/property_inspector/editor_inspector.hpp"
#include "gobot/editor/property_inspector/editor_property.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/editor/property_inspector/editor_property_primitives.hpp"
#include "gobot/editor/imgui/type_icons.hpp"
#include "gobot/type_categroy.hpp"
#include "imgui_stdlib.h"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"
#include "imgui.h"
#include "imgui_internal.h"


namespace gobot {


Ref<EditorInspectorPlugin> EditorInspector::s_inspector_plugins[MAX_PLUGINS];
int EditorInspector::s_inspector_plugin_count = 0;

EditorInspector::EditorInspector(Variant& variant)
    : cache_(variant)
{
    // check variant has name
    auto name_property = cache_.type.get_property("name");
    if (name_property.is_valid()) {
        property_name_ = new PropertyDataModel(cache_, name_property);
    }
}

EditorInspector::~EditorInspector() {
    if (property_name_)
        delete property_name_;
}


void EditorInspector::AddInspectorPlugin(const Ref<EditorInspectorPlugin> &plugin) {
    ERR_FAIL_COND(s_inspector_plugin_count == MAX_PLUGINS);

    for (int i = 0; i < s_inspector_plugin_count; i++) {
        if (s_inspector_plugins[i] == plugin) {
            return; //already exists
        }
    }
    s_inspector_plugins[s_inspector_plugin_count++] = plugin;
}

void EditorInspector::RemoveInspectorPlugin(const Ref<EditorInspectorPlugin> &plugin) {
    ERR_FAIL_COND(s_inspector_plugin_count == MAX_PLUGINS);

    int idx = -1;
    for (int i = 0; i < s_inspector_plugin_count; i++) {
        if (s_inspector_plugins[i] == plugin) {
            idx = i;
            break;
        }
    }

    ERR_FAIL_COND_MSG(idx == -1, "Trying to remove nonexistent inspector plugin.");
    for (int i = idx; i < s_inspector_plugin_count - 1; i++) {
        s_inspector_plugins[i] = s_inspector_plugins[i + 1];
    }
    s_inspector_plugins[s_inspector_plugin_count - 1] = Ref<EditorInspectorPlugin>();

    s_inspector_plugin_count--;
}

void EditorInspector::CleanupPlugins() {
    for (int i = 0; i < s_inspector_plugin_count; i++) {
        s_inspector_plugins[i].Reset();
    }
    s_inspector_plugin_count = 0;
}

bool EditorInspector::GeneraPropertyInspector() {
    if (s_inspector_plugin_count == 0) {
        return false;
    }

    return false;
}

void EditorInspector::OnImGuiContent() {
    ImGui::TextUnformatted(GetTypeIcon(cache_.type));


    if (property_name_) {
        ImGui::SameLine();

        auto str = property_name_->GetValue().to_string();
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() - 60);
        if (ImGui::InputText(fmt::format("##{}", property_name_->GetPropertyName()).c_str(), &str)) {
            property_name_->SetValue(String::fromStdString(str));
        }

        ImGui::SameLine(ImGui::GetWindowWidth() - 30);
        if (ImGui::Button(ICON_MDI_COGS)) {
            ImGui::OpenPopup("Inspector setting");
        }
        if (ImGui::BeginPopup("Inspector setting"))
        {
            if (ImGui::Button("Expand All")) {
                // TODO(wqq)
            }
            if (ImGui::Button("Collapse All")) {
                // TODO(wqq)
            }
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Copy Properties")) {
                // TODO(wqq)
            }
            if (ImGui::Button("Paste Properties")) {
                // TODO(wqq)
            }

            ImGui::EndPopup();
        }
    }

    static ImGuiTextFilter filter;
    filter.Draw(ICON_MDI_MAGNIFY "Filter", ImGui::GetWindowWidth() - 60);


}


//////////////////////////////////////////////////////////////////////////////

bool EditorInspectorDefaultPlugin::CanHandle(VariantCache& variant_cache) {
    return true; // Can handle everything.
}

//bool EditorInspectorDefaultPlugin::ParseProperty(VariantCache& variant_cache, PropertyDataModel* parent_data) {
//    auto editor = GetEditorForProperty(std::move(data_model));
//    if (editor) {
//        AddPropertyEditor(std::move(editor));
//    }
//    return false;
//}
//
//std::unique_ptr<EditorProperty> EditorInspectorDefaultPlugin::GetEditorForProperty(VariantCache& variant_cache,
//                                                                                   const Type& type) {
//    switch (GetTypeCategory(variant_cache.type)) {
//        case TypeCategory::Bool: {
//            auto* editor = new EditorPropertyBool(std::move(data_model));
//            return editor;
//        } break;
//        case TypeCategory::UInt8:
//            break;
//        case TypeCategory::UInt16:
//            break;
//        case TypeCategory::UInt32:
//            break;
//        case TypeCategory::UInt64:
//            break;
//        case TypeCategory::Int8:
//            break;
//        case TypeCategory::Int16:
//            break;
//        case TypeCategory::Int32:
//            break;
//        case TypeCategory::Int64:
//            break;
//        case TypeCategory::Float:
//            break;
//        case TypeCategory::Double:
//            break;
//        case TypeCategory::Enum:
//            break;
//        case TypeCategory::String:
//            break;
//        case TypeCategory::NodePath:
//            break;
//        case TypeCategory::Color:
//            break;
//        case TypeCategory::Vector2f:
//            break;
//        case TypeCategory::Vector2d:
//            break;
//        case TypeCategory::Vector3f:
//            break;
//        case TypeCategory::Vector3d:
//            break;
//        case TypeCategory::Vector4f:
//            break;
//        case TypeCategory::Vector4d:
//            break;
//        case TypeCategory::Quaternionf:
//            break;
//        case TypeCategory::Quaterniond:
//            break;
//        case TypeCategory::VectorXf:
//            break;
//        case TypeCategory::VectorXd:
//            break;
//        case TypeCategory::Matrix2f:
//            break;
//        case TypeCategory::Matrix2d:
//            break;
//        case TypeCategory::Matrix3f:
//            break;
//        case TypeCategory::Matrix3d:
//            break;
//        case TypeCategory::MatrixXf:
//            break;
//        case TypeCategory::MatrixXd:
//            break;
//        case TypeCategory::Ref:
//            break;
//        case TypeCategory::Array:
//            break;
//        case TypeCategory::Dictionary:
//            break;
//        case TypeCategory::Compound:
//            break;
//        case TypeCategory::Unsupported:
//            break;
//    }
//}

}