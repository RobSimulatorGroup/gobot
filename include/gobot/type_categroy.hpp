/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-11
*/

#pragma once

#include "gobot/core/types.hpp"
#include "gobot/core/color.hpp"
#include "gobot/scene/node_path.hpp"
#include "gobot/core/rid.hpp"
#include "gobot/core/math/geometry.hpp"
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
    ObjectID,
    RID,
    RenderRID,

    // math type
    Vector2i,
    Vector2f,
    Vector2d,
    Vector3i,
    Vector3f,
    Vector3d,
    Vector4i,
    Vector4f,
    Vector4d,
    VectorXi,
    VectorXf,
    VectorXd,
    Matrix2i,
    Matrix2f,
    Matrix2d,
    Matrix3i,
    Matrix3f,
    Matrix3d,
    MatrixXi,
    MatrixXf,
    MatrixXd,
    Quaternionf,
    Quaterniond,
    Isometry2f,
    Isometry2d,
    Isometry3f,
    Isometry3d,
    Affine2f,
    Affine2d,
    Affine3f,
    Affine3d,
    Projective2f,
    Projective2d,
    Projective3f,
    Projective3d,

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
    } else if (type == Type::get<ObjectID>()) {
        return TypeCategory::ObjectID;
    } else if (type == Type::get<RID>()) {
        return TypeCategory::RID;
    }

    // math related
    else if (type == Type::get<Vector2i>()) {
        return TypeCategory::Vector2i;
    } else if (type == Type::get<Vector2f>()) {
        return TypeCategory::Vector2f;
    } else if (type == Type::get<Vector2d>()) {
        return TypeCategory::Vector2d;
    } else if (type == Type::get<Vector3i>()) {
        return TypeCategory::Vector3i;
    } else if (type == Type::get<Vector3f>()) {
        return TypeCategory::Vector3f;
    } else if (type == Type::get<Vector3d>()) {
        return TypeCategory::Vector3d;
    } else if (type == Type::get<Vector4i>()) {
        return TypeCategory::Vector4i;
    } else if (type == Type::get<Vector4f>()) {
        return TypeCategory::Vector4f;
    } else if (type == Type::get<Vector4d>()) {
        return TypeCategory::Vector4d;
    } else if (type == Type::get<VectorXi>()) {
        return TypeCategory::VectorXi;
    } else if (type == Type::get<VectorXf>()) {
        return TypeCategory::VectorXf;
    } else if (type == Type::get<VectorXd>()) {
        return TypeCategory::VectorXd;
    } else if (type == Type::get<Matrix2i>()) {
        return TypeCategory::Matrix2i;
    } else if (type == Type::get<Matrix2f>()) {
        return TypeCategory::Matrix2f;
    } else if (type == Type::get<Matrix2d>()) {
        return TypeCategory::Matrix2d;
    } else if (type == Type::get<Matrix3i>()) {
        return TypeCategory::Matrix3i;
    } else if (type == Type::get<Matrix3f>()) {
        return TypeCategory::Matrix3f;
    } else if (type == Type::get<Matrix3d>()) {
        return TypeCategory::Matrix3d;
    } else if (type == Type::get<MatrixXi>()) {
        return TypeCategory::MatrixXi;
    } else if (type == Type::get<MatrixXf>()) {
        return TypeCategory::MatrixXf;
    } else if (type == Type::get<MatrixXd>()) {
        return TypeCategory::MatrixXd;
    } else if (type == Type::get<Quaternionf>()) {
        return TypeCategory::Quaternionf;
    } else if (type == Type::get<Quaterniond>()) {
        return TypeCategory::Quaterniond;
    } else if (type == Type::get<Isometry2f>()) {
        return TypeCategory::Isometry2f;
    } else if (type == Type::get<Isometry2d>()) {
        return TypeCategory::Isometry2d;
    } else if (type == Type::get<Isometry3f>()) {
        return TypeCategory::Isometry3f;
    } else if (type == Type::get<Isometry3d>()) {
        return TypeCategory::Isometry3d;
    } else if (type == Type::get<Affine2f>()) {
        return TypeCategory::Affine2f;
    } else if (type == Type::get<Affine2d>()) {
        return TypeCategory::Affine2d;
    } else if (type == Type::get<Affine3f>()) {
        return TypeCategory::Affine3f;
    } else if (type == Type::get<Affine3d>()) {
        return TypeCategory::Affine3d;
    } else if (type == Type::get<Projective2f>()) {
        return TypeCategory::Projective2f;
    } else if (type == Type::get<Projective2d>()) {
        return TypeCategory::Projective2d;
    } else if (type == Type::get<Projective3f>()) {
        return TypeCategory::Projective3f;
    } else if (type == Type::get<Projective3d>()) {
        return TypeCategory::Projective3d;
    }

        // complex type
    else if (type.is_wrapper()) {
        if (type.get_wrapper_holder_type() == WrapperHolderType::Ref) {
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