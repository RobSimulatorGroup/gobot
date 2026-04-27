/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
*/

#pragma once

#include "gobot/editor/property_inspector/editor_property_builtin.hpp"

namespace gobot {

class EditorPropertyResource : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyResource, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

    void End() override;
};

}
