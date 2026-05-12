/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-11-6
 * SPDX-License-Identifier: Apache-2.0
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

    virtual bool ParseProperty(std::unique_ptr<VariantDataModel> variant_data) = 0;

    void AddEditor(ImGuiNode* editor) {
        added_editors_.emplace_back(editor);
    }

    const std::vector<ImGuiNode*>& GetAddEditors() const {
        return added_editors_;
    }

    std::vector<ImGuiNode*>& GetAddEditors() {
        return added_editors_;
    }


public:
    std::vector<ImGuiNode*> added_editors_{};
};


class PropertyDataModel;

class EditorInspector : public ImGuiNode {
    GOBCLASS(EditorInspector, ImGuiNode)
public:
    EditorInspector(Variant& variant);

    ~EditorInspector() override;

    void PrintAllProperties();

    void InitializeEditors();

    static void AddInspectorPlugin(const Ref<EditorInspectorPlugin> &plugin);

    static void RemoveInspectorPlugin(const Ref<EditorInspectorPlugin> &plugin);

    static void CleanupPlugins();

    bool Begin() override;

    void OnImGuiContent() override;

    void End() override;

    VariantCache& GetVariantCache();

    PropertyDataModel* GetNameProperty();

private:
    enum {
        MAX_PLUGINS = 1024
    };

    static Ref<EditorInspectorPlugin> s_inspector_plugins[MAX_PLUGINS];
    static int s_inspector_plugin_count;

    VariantCache cache_;
    PropertyDataModel* property_name_{nullptr};

    // base class --> derived class
    std::vector<Type> inheritance_chain_{};
    std::map<Type, std::vector<Property>> properties_map_{};
};


/////////////////////////////////////////////////////////////////////////////


class EditorInspectorDefaultPlugin : public EditorInspectorPlugin {
    GOBCLASS(EditorInspectorDefaultPlugin, EditorInspectorPlugin);
public:
    EditorInspectorDefaultPlugin() = default;

    bool CanHandle(VariantCache& variant_cache) override;

    bool ParseProperty(std::unique_ptr<VariantDataModel> variant_data) override;

    static ImGuiNode* GetEditorForProperty(std::unique_ptr<VariantDataModel> variant_data);

};


}