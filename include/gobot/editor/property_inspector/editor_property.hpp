/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-28
*/

#pragma once

#include "gobot/core/object.hpp"
#include "gobot/core/types.hpp"
#include "gobot/editor/imgui/imgui_utilities.hpp"

namespace gobot {

class EditorProperty {
public:
    EditorProperty(Variant& variant, const Property& property)
    : variant_(variant),
    property_(property),
    property_cache_(variant_)
    {
        property_cache_.property_name = property_.get_name().data();
        property_cache_.read_only = property_.is_readonly();
    }

    virtual void OnDataImGui() = 0;

    void OnImGui() {
        ImGuiUtilities::BeginPropertyGrid(property_cache_.property_name.data());
        OnDataImGui();
        ImGuiUtilities::EndPropertyGrid();
    };


protected:
    struct PropertyCache {
        Instance instance;
        Type type;
        Object* object{nullptr};
        std::string property_name{};
        bool read_only{false};

        explicit PropertyCache(Instance _instance)
                : instance(_instance.get_type().get_raw_type().is_wrapper() ? _instance.get_wrapped_instance() : _instance),
                  type(_instance.get_type().get_raw_type()),
                  object(instance.try_convert<Object>())
        {
        }
    };

    Property property_;
    Variant& variant_;

    PropertyCache property_cache_;
};

}