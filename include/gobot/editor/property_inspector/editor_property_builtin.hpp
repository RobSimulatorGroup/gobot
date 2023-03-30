/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-30
*/

#pragma once

#include "gobot/editor/property_inspector/editor_property.hpp"

namespace gobot {

template<typename T>
class EditorBuiltInProperty : public EditorProperty {
public:
    using BaseClass = EditorBuiltInProperty<T>;

    using EditorProperty::EditorProperty;



protected:

    T data_;
};

}
