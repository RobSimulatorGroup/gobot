/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-28
*/

#include "gobot/editor/property_inspector/editor_inspector.hpp"
#include "gobot/editor/property_inspector/editor_property.hpp"
#include "gobot/scene/imgui_custom_node.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/editor/property_inspector/editor_property_primitives.hpp"
#include "gobot/editor/property_inspector/editor_property_math.hpp"
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


    // collect all properties
    for (const auto& base: cache_.type.get_base_classes()) {
        inheritance_chain_.emplace_back(base);
    }
    inheritance_chain_.emplace_back(cache_.type);
    std::reverse(inheritance_chain_.begin(), inheritance_chain_.end());

    for (const auto& type :  inheritance_chain_) {
        properties_map_.insert({type, {}});
        for (const auto& property : type.get_properties(rttr::filter_item::public_access |
                                                        rttr::filter_item::declared_only |
                                                        rttr::filter_item::static_item |
                                                        rttr::filter_item::instance_item)) {
            if (property == name_property) {
                continue;
            }
            properties_map_.at(type).emplace_back(property);
        }
    }

#ifdef GOBOT_DEBUG
    PrintAllProperties();
#endif

    InitializeEditors();
}

EditorInspector::~EditorInspector() {
    delete property_name_;
}

void EditorInspector::PrintAllProperties() {
    for (const auto& type : inheritance_chain_) {
        LOG_INFO("Type: {}", type.get_name().data());
        for (const auto& prop : properties_map_.at(type)) {
            LOG_ERROR("-- {}", prop.get_name().data());
        }
    }
}

void EditorInspector::InitializeEditors() {
    if (s_inspector_plugin_count == 0) {
        return;
    }

    std::vector<Ref<EditorInspectorPlugin>> valid_plugins;

    for (int i = s_inspector_plugin_count - 1; i >= 0; i--) { //start by last, so lastly added can override newly added
        if (!s_inspector_plugins[i]->CanHandle(cache_)) {
            continue;
        }
        valid_plugins.push_back(s_inspector_plugins[i]);
    }

    for (const auto& type: inheritance_chain_) {
        AddChild(ImGuiCustomNode::New<ImGuiCustomNode>([&type]() {
            ImGui::SeparatorText(fmt::format("{} {}", GetTypeIcon(type), type.get_name().data()).c_str());
        }));

        for (const auto& prop : properties_map_.at(type)) {
            for (Ref<EditorInspectorPlugin> &valid_plugin : valid_plugins) {
                bool exclusive = valid_plugin->ParseProperty(std::make_unique<PropertyDataModel>(cache_, prop));
                for (const auto& editor : valid_plugin->GetAddEditors()) {
                    AddChild(editor);
                }
                valid_plugin->GetAddEditors().clear();
                if (exclusive) {
                    break;
                }
            }
        }
    }

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

bool EditorInspector::Begin() {
    ImGui::BeginChild(fmt::format("##{}", fmt::ptr(this)).c_str());
    return true;
}

void EditorInspector::End() {
    ImGui::EndChild();
}

VariantCache& EditorInspector::GetVariantCache() {
    return cache_;
}

PropertyDataModel* EditorInspector::GetNameProperty() {
    return property_name_;
}

void EditorInspector::OnImGuiContent() {


}


//////////////////////////////////////////////////////////////////////////////

bool EditorInspectorDefaultPlugin::CanHandle(VariantCache& variant_cache) {
    return true; // Can handle everything.
}

bool EditorInspectorDefaultPlugin::ParseProperty(std::unique_ptr<VariantDataModel> variant_data) {
    auto editor = GetEditorForProperty(std::move(variant_data));
    if (editor) {
        AddEditor(editor);
    }
    return false;
}

ImGuiNode* EditorInspectorDefaultPlugin::GetEditorForProperty(std::unique_ptr<VariantDataModel> variant_data) {
    auto type_category = GetTypeCategory(variant_data->GetValueType());
    switch (type_category) {
        case TypeCategory::Bool: {
            auto* editor = EditorPropertyBool::New<EditorPropertyBool>(type_category, std::move(variant_data));
            return editor;
        } break;
        case TypeCategory::UInt8:
        case TypeCategory::UInt16:
        case TypeCategory::UInt32:
        case TypeCategory::UInt64:
        case TypeCategory::Int8:
        case TypeCategory::Int16:
        case TypeCategory::Int32:
        case TypeCategory::Int64: {
            auto* editor = Object::New<EditorPropertyInteger>(type_category, std::move(variant_data));
            return editor;
        } break;
        case TypeCategory::Float:
        case TypeCategory::Double: {
            auto* editor = Object::New<EditorPropertyFloat>(type_category, std::move(variant_data));
            return editor;
        } break;
        case TypeCategory::Enum: {
            if ((dynamic_cast<PropertyDataModel*>(variant_data.get())->GetPropertyInfo().hint == PropertyHint::Flags)) {
                auto* editor = Object::New<EditorPropertyFlags>(type_category, std::move(variant_data));
                return editor;
            } else {
                auto* editor = Object::New<EditorPropertyEnum>(type_category, std::move(variant_data));
                return editor;
            }
        } break;
        case TypeCategory::String: {
            if ((dynamic_cast<PropertyDataModel*>(variant_data.get())->GetPropertyInfo().hint == PropertyHint::MultilineText)) {
                auto* editor = Object::New<EditorPropertyMultilineText>(type_category, std::move(variant_data));
                return editor;
            } else {
                auto* editor = Object::New<EditorPropertyText>(type_category, std::move(variant_data));
                return editor;
            }
        } break;
        case TypeCategory::NodePath: {
            auto* editor = Object::New<EditorPropertyNodePath>(type_category, std::move(variant_data));
            return editor;
        }
        case TypeCategory::Color: {
            auto* editor = Object::New<EditorPropertyColor>(type_category, std::move(variant_data));
            return editor;
        } break;
        case TypeCategory::ObjectID:
            break;
        case TypeCategory::RID:
            break;
        case TypeCategory::RenderRID:
            break;
        case TypeCategory::Vector2i:
        case TypeCategory::Vector2f:
        case TypeCategory::Vector2d: {
            auto* editor = Object::New<EditorPropertyVector2>(type_category, std::move(variant_data));
            return editor;
        } break;
        case TypeCategory::Vector3i:
        case TypeCategory::Vector3f:
        case TypeCategory::Vector3d: {
            auto* editor = Object::New<EditorPropertyVector3>(type_category, std::move(variant_data));
            return editor;
        } break;
        case TypeCategory::Vector4i:
        case TypeCategory::Vector4f:
        case TypeCategory::Vector4d: {
            auto* editor = Object::New<EditorPropertyVector4>(type_category, std::move(variant_data));
            return editor;
        } break;

        case TypeCategory::Matrix2i:
        case TypeCategory::Matrix2f:
        case TypeCategory::Matrix2d: {
            auto* editor = Object::New<EditorPropertyMatrix2>(type_category, std::move(variant_data));
            return editor;
        } break;
        case TypeCategory::Matrix3i:
        case TypeCategory::Matrix3f:
        case TypeCategory::Matrix3d: {
            auto* editor = Object::New<EditorPropertyMatrix3>(type_category, std::move(variant_data));
            return editor;
        } break;
        case TypeCategory::VectorXi:
        case TypeCategory::VectorXf:
        case TypeCategory::VectorXd: {
            auto* editor = Object::New<EditorPropertyVectorX>(type_category, std::move(variant_data));
            return editor;
        } break;
        case TypeCategory::MatrixXi:
        case TypeCategory::MatrixXf:
        case TypeCategory::MatrixXd: {
            auto* editor = Object::New<EditorPropertyMatrixX>(type_category, std::move(variant_data));
            return editor;
        } break;
        case TypeCategory::Quaternionf:
        case TypeCategory::Quaterniond: {
            auto* editor = Object::New<EditorPropertyQuaternion>(type_category, std::move(variant_data));
            return editor;
        } break;
        case TypeCategory::Isometry2f:
        case TypeCategory::Isometry2d:
        case TypeCategory::Affine2f:
        case TypeCategory::Affine2d:
        case TypeCategory::Projective2f:
        case TypeCategory::Projective2d: {
            auto* editor = Object::New<EditorPropertyTransform2>(type_category, std::move(variant_data));
            return editor;
        } break;
        case TypeCategory::Isometry3f:
        case TypeCategory::Isometry3d:
        case TypeCategory::Affine3f:
        case TypeCategory::Affine3d:
        case TypeCategory::Projective3f:
        case TypeCategory::Projective3d: {
            auto* editor = Object::New<EditorPropertyTransform3>(type_category, std::move(variant_data));
            return editor;
        } break;
        case TypeCategory::Ref:
            break;
        case TypeCategory::Array:
            break;
        case TypeCategory::Dictionary:
            break;
        case TypeCategory::Compound:
            break;
        case TypeCategory::Unsupported:
            break;
        case TypeCategory::Invalid:
            break;
    }

    return nullptr;
}

}