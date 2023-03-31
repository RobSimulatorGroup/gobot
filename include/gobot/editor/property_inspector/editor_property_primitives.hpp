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

class EditorPropertyNil : public EditorBuiltInProperty<nullptr_t> {
public:

    using BaseClass::BaseClass;

    void OnDataImGui() override;

};

class EditorPropertyBool : public EditorBuiltInProperty<bool> {
public:
    using BaseClass::BaseClass;

    void OnDataImGui() override;

};

template<typename Flags>
class EditorPropertyFlags : public EditorBuiltInProperty<Flags> {
public:
    using Base = EditorBuiltInProperty<Flags>;
    using Base::Base;

    void OnDataImGui() override;

};


class EditorPropertyText : public EditorBuiltInProperty<String> {
public:
    using BaseClass::BaseClass;

    void OnDataImGui() override;

};

class EditorPropertyMultilineText : public EditorBuiltInProperty<String> {
public:
    using BaseClass::BaseClass;

    void OnDataImGui() override;

};

class EditorPropertyPath : public EditorBuiltInProperty<String> {
public:
    using BaseClass::BaseClass;

    void OnDataImGui() override;
};

template<typename Integer>
class EditorPropertyInteger : public EditorBuiltInProperty<Integer> {
public:
    EditorPropertyInteger();

    void OnDataImGui() override;
};

template<typename Float>
class EditorPropertyFloat : public EditorBuiltInProperty<Float> {
public:
    EditorPropertyFloat();

    void OnDataImGui() override;
};

class EditorPropertyRID : public EditorBuiltInProperty<RID> {
public:
    EditorPropertyRID();

    void OnDataImGui() override;
};


}