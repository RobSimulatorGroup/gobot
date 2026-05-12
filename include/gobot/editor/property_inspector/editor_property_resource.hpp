/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
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
