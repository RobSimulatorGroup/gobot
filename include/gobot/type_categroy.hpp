/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-11
*/

#pragma once

#include "gobot/core/types.hpp"
#include "gobot/scene/node_path.hpp"
#include "gobot/log.hpp"

namespace gobot {

enum class TypeCategory {
    Invalid = 0,
    // built-in type
    Bool,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float,
    Double,
    Enum,
    String,
    NodePath,
    Color,

    // math type
    Vector2f,
    Vector2d,
    Vector3f,
    Vector3d,
    Vector4f,
    Vector4d,
    Quaternionf,
    Quaterniond,
    VectorXf,
    VectorXd,
    Matrix2f,
    Matrix2d,
    Matrix3f,
    Matrix3d,
    MatrixXf,
    MatrixXd,

    // complex type
    Ref,
    Array,
    Dictionary,
    Compound,
    Unsupported
};

inline TypeCategory GetTypeCategory(const Type& type) {
    if (!type.is_valid()) {
        return TypeCategory::Invalid;
    }
    // built-in type
    if (type.is_arithmetic()) {
        if (type == Type::get<bool>()) {
            return TypeCategory::Bool;
        } else if (type == Type::get<std::uint8_t>()) {
            return TypeCategory::UInt8;
        } else if (type == Type::get<std::uint16_t>()) {
            return TypeCategory::UInt16;
        } else if (type == Type::get<std::uint32_t>()) {
            return TypeCategory::UInt32;
        } else if (type == Type::get<std::uint64_t>()) {
            return TypeCategory::UInt64;
        } else if (type == Type::get<std::int8_t>()) {
            return TypeCategory::Int8;
        } else if (type == Type::get<std::int16_t>()) {
            return TypeCategory::Int16;
        } else if (type == Type::get<std::int32_t>()) {
            return TypeCategory::Int32;
        } else if (type == Type::get<std::int64_t>()) {
            return TypeCategory::Int64;
        } else if (type == Type::get<float>()) {
            return TypeCategory::Float;
        } else if (type == Type::get<double>()) {
            return TypeCategory::Double;
        }
    } else if (type.is_enumeration()) {
        return TypeCategory::Enum;
    } else if (type == Type::get<String>()) {
        return TypeCategory::String;
    } else if (type == Type::get<NodePath>()) {
        return TypeCategory::NodePath;
    } else if (type == Type::get<Color>()) {
        return TypeCategory::Color;
    }
    // TODO(wqq): math types

    else if (type.is_wrapper()) {
        if (type.get_wrapper_holder_type() == rttr::wrapper_holder_type::Ref) {
            return TypeCategory::Ref;
        }
        // Other type is Unsupported
        LOG_WARN("The wrapper type:{} is Unsupported", type.get_name().data());
        return TypeCategory::Unsupported;
    } else if (type.is_sequential_container()) {
        return TypeCategory::Array;
    } else if (type.is_associative_container()) {
        return TypeCategory::Dictionary;
    } else if (!type.get_properties().empty()) {
        return TypeCategory::Compound;
    }
    LOG_WARN("The type:{} is Unsupported", type.get_name().data());
    return TypeCategory::Unsupported;
}

}