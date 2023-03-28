/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-28
*/

#pragma once

#include "gobot/editor/property_inspector/editor_inspector.hpp"
#include "gobot/editor/property_inspector/editor_property.hpp"

namespace gobot {

class EditorPropertyNil : public EditorProperty {
    GOBCLASS(EditorPropertyNil, EditorProperty)
public:
    EditorPropertyNil();

    void OnImGui() override;

};

class EditorPropertyCheck : public EditorProperty {
    GOBCLASS(EditorPropertyCheck, EditorProperty)
public:
    EditorPropertyCheck();

    void OnImGui() override;

};


class EditorPropertyFlags : public EditorProperty {
    GOBCLASS(EditorPropertyFlags, EditorProperty)
public:
    EditorPropertyFlags();

    void OnImGui() override;

};


class EditorPropertyText : public EditorProperty {
    GOBCLASS(EditorPropertyText, EditorProperty)
public:
    EditorPropertyText();

    void OnImGui() override;

};

class EditorPropertyMultilineText : public EditorProperty {
    GOBCLASS(EditorPropertyMultilineText, EditorProperty)
public:
    EditorPropertyMultilineText();

    void OnImGui() override;

};

class EditorPropertyPath : public EditorProperty {
    GOBCLASS(EditorPropertyPath, EditorProperty)
public:
    EditorPropertyPath();

    void OnImGui() override;
};

class EditorPropertyInteger : public EditorProperty {
    GOBCLASS(EditorPropertyInteger, EditorProperty)
public:
    EditorPropertyInteger();

    void OnImGui() override;
};

class EditorPropertyFloat : public EditorProperty {
    GOBCLASS(EditorPropertyFloat, EditorProperty)
public:
    EditorPropertyFloat();

    void OnImGui() override;
};

class EditorPropertyRID : public EditorProperty {
    GOBCLASS(EditorPropertyRID, EditorProperty)
public:
    EditorPropertyRID();

    void OnImGui() override;
};

/////////////////////////////////////////////////////////////////////////////
class EditorInspectorDefaultPlugin : public EditorInspectorPlugin {
    GOBCLASS(EditorInspectorDefaultPlugin, EditorInspectorPlugin);
public:
    bool CanHandle(Instance instance) override;

    bool ParseProperty(Instance instance) override;

};


}