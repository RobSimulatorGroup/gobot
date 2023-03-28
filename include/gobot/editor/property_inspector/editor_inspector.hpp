/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-6
*/


#pragma once

#include "gobot/editor/property_inspector/editor_inspector.hpp"
#include "gobot/core/ref_counted.hpp"

namespace gobot {

class EditorInspectorPlugin : public RefCounted {
    GOBCLASS(EditorInspectorPlugin, RefCounted)
public:
    virtual bool CanHandle(Instance instance) = 0;

    virtual bool ParseProperty(Instance instance) = 0;

};


class EditorInspector : public Object {
    GOBCLASS(EditorInspector, Object)
public:
    EditorInspector(Variant& variant);

    static void AddInspectorPlugin(const Ref<EditorInspectorPlugin> &plugin);

    static void RemoveInspectorPlugin(const Ref<EditorInspectorPlugin> &plugin);

    static void CleanupPlugins();

    void OnImGui();

private:
    enum {
        MAX_PLUGINS = 1024
    };

    static Ref<EditorInspectorPlugin> s_inspector_plugins[MAX_PLUGINS];
    static int s_inspector_plugin_count;

    Variant& variant_;

    // cache for variant
    struct Cache {
        String name;
        Type type;

        Cache(Type _type)
            : type(std::move(_type))
        {
        }
    };

    Cache cache_;
};


}