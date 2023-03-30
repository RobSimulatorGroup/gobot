/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-28
*/

#include "gobot/editor/property_inspector/editor_inspector.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/editor/imgui/type_icons.hpp"
#include "imgui.h"


namespace gobot {


Ref<EditorInspectorPlugin> EditorInspector::s_inspector_plugins[MAX_PLUGINS];
int EditorInspector::s_inspector_plugin_count = 0;


EditorInspector::Cache::Cache(Instance _instance)
    : instance(_instance.get_type().get_raw_type().is_wrapper() ? _instance.get_wrapped_instance() : _instance),
      type(_instance.get_type().get_raw_type()),
      object(instance.try_convert<Object>())
{
    auto name_property = type.get_property("name");
    if (name_property.is_valid()) {
        name = name_property.get_value(instance).to_string();
    }
}

EditorInspector::EditorInspector(Variant& variant)
    : variant_(variant),
      cache_(variant_)
{
    // check variant has name

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

void EditorInspector::OnImGui() {
    ImGui::TextUnformatted(GetTypeIcon(cache_.type));
    ImGui::SameLine();
    static char str0[128] = "Hello, world!";
    if (ImGui::InputText("##name", str0, IM_ARRAYSIZE(str0))) {
        auto name_property = cache_.type.get_property("name");
        String str = str0;
        cache_.name = str0;
        name_property.set_value(cache_.instance, str);
    }

}


}