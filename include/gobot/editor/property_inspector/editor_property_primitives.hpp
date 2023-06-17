/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-28
*/

#pragma once

#include "gobot/editor/property_inspector/editor_property_builtin.hpp"
#include "gobot/core/rid.hpp"

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
    EditorPropertyFlags(TypeCategory type_category, std::unique_ptr<VariantDataModel> variant_data_model);

    void OnImGuiContent() override;

private:
    enum FlagsUnderlyingType {
        UInt8,
        UInt16,
        UInt32,
        Int8,
        Int16,
        Int32,
    };

    Enumeration enumeration_;
    FlagsUnderlyingType underlying_type_;
    int int_data_{0};
    unsigned int uint_data_{0};
    std::vector<std::pair<std::string_view, Variant>> names_;
};


class EditorPropertyEnum : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyEnum, EditorBuiltInProperty)
public:
    EditorPropertyEnum(TypeCategory type_category, std::unique_ptr<VariantDataModel> variant_data_model);

    void OnImGuiContent() override;

private:
    Enumeration enumeration_;
    std::vector<const char*> names_;
    std::unordered_map<std::string, int> names_map_;
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

class EditorPropertyColor : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyColor, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

private:
    float color_[4];
};

class EditorPropertyObjectID : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyObjectID, EditorBuiltInProperty)
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


}