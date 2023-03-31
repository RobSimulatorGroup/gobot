/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-31
*/

#pragma once

#include "gobot/core/types.hpp"
#include "gobot/core/object.hpp"
#include "gobot/log.hpp"
#include "gobot/editor/property_inspector/variant_cache.hpp"

namespace gobot {

enum class DataModelType {
    Property,
    SequenceContainer,
    AssociativeContainer,
    Function
};


class VariantDataModel {
public:
    explicit VariantDataModel(VariantCache& variant_cache)
        : variant_cache_(variant_cache)
    {
    }

    [[nodiscard]] virtual DataModelType GetDataModelType() const = 0;

    [[nodiscard]] virtual const Type& GetValueType() const = 0;

    [[nodiscard]] const Type& GetHolderType() const { return variant_cache_.type; }

protected:

    VariantCache& variant_cache_;
};

class PropertyDataModel : public VariantDataModel {
public:
    PropertyDataModel(VariantCache& variant, const Property& property);

    [[nodiscard]] DataModelType GetDataModelType() const override { return DataModelType::Property; };

    [[nodiscard]] const Type& GetValueType() const override;

    [[nodiscard]] const String& GetPropertyName() const;

    [[nodiscard]] bool IsPropertyReadOnly() const;

    [[nodiscard]] const PropertyInfo& GetPropertyInfo() const;

    bool SetValue(Argument argument);

    Variant GetValue() const;

private:
    struct PropertyCache {
        Type property_type;
        String property_name;
        bool property_readonly;
        PropertyInfo property_info;
        PropertyCache(const Type& type, String string, bool readonly, const Variant& info)
            : property_type(type),
              property_name(std::move(string)),
              property_readonly(readonly),
              property_info(info.is_valid() ? info.get_value<PropertyInfo>() : PropertyInfo())
        {
        }
    };

    Property property_;
    PropertyCache property_cache_;
};


class SequenceContainerDataModel : public VariantDataModel {
public:
    explicit SequenceContainerDataModel(VariantCache& variant);

    [[nodiscard]] DataModelType GetDataModelType() const override { return DataModelType::SequenceContainer; };

    [[nodiscard]] const Type& GetValueType() const override;

    [[nodiscard]] std::size_t GetSize() const;

    [[nodiscard]] Variant GetValue(std::size_t index) const;

    void SetValue(std::size_t index, Argument argument);

private:
    struct SequenceContainerCache {
        VariantListView variant_list_view;
        Type value_type;
        explicit SequenceContainerCache(const VariantListView& view)
            : variant_list_view(view),
              value_type(variant_list_view.get_value_type()) {}
    };

    SequenceContainerCache sc_cache_;
};


class AssociativeContainerDataModel : public VariantDataModel {
public:
    explicit AssociativeContainerDataModel(VariantCache& variant);

    [[nodiscard]] DataModelType GetDataModelType() const override { return DataModelType::AssociativeContainer; };

    [[nodiscard]] const Type& GetValueType() const override;

    [[nodiscard]] bool IsKeyOnlyType() const;

    [[nodiscard]] const Type& GetKeyType() const;

private:
    struct AssociativeContainerCache {
        VariantMapView variant_map_view;
        bool is_key_only_type;
        Type key_type;
        Type value_type;
        explicit AssociativeContainerCache(const VariantMapView& view)
                : variant_map_view(view),
                  is_key_only_type(variant_map_view.is_key_only_type()),
                  key_type(variant_map_view.get_key_type()),
                  value_type(variant_map_view.get_value_type())

        {
        }
    };

    AssociativeContainerCache ac_cache_;
};


class FunctionDataModel : public VariantDataModel {
public:
    FunctionDataModel(VariantCache& variant, const Method& method);

    [[nodiscard]] DataModelType GetDataModelType() const override { return DataModelType::Function; };

    [[nodiscard]] const Type& GetValueType() const override;

    [[nodiscard]] const String& GetMethodName() const;

    void DoMethodCall();

private:
    struct MethodCache {
        String method_name;
        Type declaring_type;
        Type return_type;
    };

    Method method_;
    MethodCache method_cache_;
};



}
