/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-4-5
*/

#pragma once

#include "gobot/editor/property_inspector/editor_property_builtin.hpp"

namespace gobot {

class EditorPropertyVector2 : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyVector2, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

private:
};


///////////////////////////////////////////////////////////////////

class EditorPropertyVector3 : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyVector3, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

private:
};


///////////////////////////////////////////////////////////////////////


class EditorPropertyVector4 : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyVector4, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

private:
};

//////////////////////////////////////////////////////////////////////

class EditorPropertyQuaternion : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyQuaternion, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

private:
    bool free_edit_mode_{false};
};


}
