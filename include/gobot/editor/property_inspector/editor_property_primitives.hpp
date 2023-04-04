/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-28
*/

#pragma once

#include "gobot/editor/property_inspector/editor_property_builtin.hpp"
#include "gobot/core/rid.h"

namespace gobot {


class EditorPropertyBool : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyBool, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

};

class EditorPropertyInteger : public EditorBuiltInProperty {
GOBCLASS(EditorPropertyInteger, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

private:
    const float drag_speed_{0.2f};
    bool drag_clamp_{false};
};


class EditorPropertyFloat : public EditorBuiltInProperty {
GOBCLASS(EditorPropertyFloat, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

private:
    const float drag_speed_{0.005f};
    bool drag_clamp_{false};
};

class EditorPropertyFlags : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyFlags, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

};


class EditorPropertyEnum : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyEnum, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

};


class EditorPropertyText : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyText, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

};


class EditorPropertyMultilineText : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyMultilineText, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

};

class EditorPropertyPath : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyPath, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;
};


class EditorPropertyNodePath : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyNodePath, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

};

class EditorPropertyRID : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyRID, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;
};

class EditorPropertyRenderRID : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyRenderRID, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;
};


}