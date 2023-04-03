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

    void OnImGuiContent() override;

};

class EditorPropertyFlags : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyFlags, EditorBuiltInProperty)
public:

    void OnImGuiContent() override;

};


class EditorPropertyText : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyText, EditorBuiltInProperty)
public:

    void OnImGuiContent() override;

};

class EditorPropertyMultilineText : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyMultilineText, EditorBuiltInProperty)
public:

    void OnImGuiContent() override;

};

class EditorPropertyPath : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyPath, EditorBuiltInProperty)
public:
    void OnImGuiContent() override;
};

class EditorPropertyInteger : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyInteger, EditorBuiltInProperty)
public:
    EditorPropertyInteger();

    void OnImGuiContent() override;
};

class EditorPropertyFloat : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyFloat, EditorBuiltInProperty)
public:
    EditorPropertyFloat();

    void OnImGuiContent() override;
};

class EditorPropertyRID : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyRID, EditorBuiltInProperty)
public:
    EditorPropertyRID();

    void OnImGuiContent() override;
};


}