/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-6
*/


#pragma once

#include "gobot/editor/property_inspector/editor_inspector.hpp"
#include "gobot/editor/property_inspector/variant_cache.hpp"
#include "gobot/editor/property_inspector/editor_property.hpp"
#include "gobot/scene/imgui_node.hpp"
#include "gobot/core/ref_counted.hpp"

namespace gobot {

class EditorInspectorPlugin : public RefCounted {
    GOBCLASS(EditorInspectorPlugin, RefCounted)
public:
    virtual bool CanHandle(VariantCache& variant_cache) = 0;

    virtual bool ParseProperty(VariantCache& variant_cache, PropertyDataModel* parent_data) = 0;

//    void AddPropertyEditorRoot(std::unique_ptr<EditorProperty> root) {
//        if (root == nullptr) {
//            root = std::make_unique<AddEditorNode>(std::move(editor_property));
//        } else {
//
//        }
//    }

private:
    std::unique_ptr<EditorProperty> root{nullptr};
};


class PropertyDataModel;

class EditorInspector : public ImGuiNode {
    GOBCLASS(EditorInspector, ImGuiNode)
public:
    EditorInspector(Variant& variant);

    virtual ~EditorInspector();

    static void AddInspectorPlugin(const Ref<EditorInspectorPlugin> &plugin);

    static void RemoveInspectorPlugin(const Ref<EditorInspectorPlugin> &plugin);

    static void CleanupPlugins();

    bool GeneraPropertyInspector();

    void OnImGuiContent();

private:
    enum {
        MAX_PLUGINS = 1024
    };

    static Ref<EditorInspectorPlugin> s_inspector_plugins[MAX_PLUGINS];
    static int s_inspector_plugin_count;

    VariantCache cache_;
    PropertyDataModel* property_name_{nullptr};
};


/////////////////////////////////////////////////////////////////////////////


class EditorInspectorDefaultPlugin : public EditorInspectorPlugin {
    GOBCLASS(EditorInspectorDefaultPlugin, EditorInspectorPlugin);
public:
    bool CanHandle(VariantCache& variant_cache) override;

//    bool ParseProperty(std::unique_ptr<VariantDataModel> data_model) override;

    static std::unique_ptr<EditorProperty> GetEditorForProperty(VariantCache& variant_cache);

};


}