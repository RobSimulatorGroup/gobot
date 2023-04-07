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

    std::variant<Vector2i, Vector2f, Vector2d> data_;
};


///////////////////////////////////////////////////////////////////

class EditorPropertyVector3 : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyVector3, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

private:
    std::variant<Vector3i, Vector3f, Vector3d> data_;
};


///////////////////////////////////////////////////////////////////////


class EditorPropertyVector4 : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyVector4, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

private:
    std::variant<Vector4i, Vector4f, Vector4d> data_;
};

//////////////////////////////////////////////////////////////////////

class EditorPropertyQuaternion : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyQuaternion, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

private:
    bool free_edit_mode_{false};
    std::variant<Quaternionf, Quaterniond> data_;
};

//////////////////////////////////////////////////////////////////////

class EditorPropertyMatrix2 : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyMatrix2, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

private:
    std::variant<Matrix2i, Matrix2f, Matrix2d> data_;
};

//////////////////////////////////////////////////////////////////////

class EditorPropertyMatrix3 : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyMatrix3, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

private:
    std::variant<Matrix3i, Matrix3f, Matrix3d> data_;
};

//////////////////////////////////////////////////////////////////////

class EditorPropertyVectorX : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyVectorX, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

private:
    std::variant<VectorXi, VectorXf, VectorXd> data_;
    int rows_{0};
};


//////////////////////////////////////////////////////////////////////

class EditorPropertyMatrixX : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyMatrixX, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

private:
    std::variant<MatrixXi, MatrixXf, MatrixXd> data_;
    int rows_{0};
    int columns_{0};
};


//////////////////////////////////////////////////////////////////////

class EditorPropertyTransform2 : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyTransform2, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

private:
    std::variant<Isometry2f, Isometry2d, Affine2f, Affine2d, Projective2f, Projective2d> data_;
};

//////////////////////////////////////////////////////////////////////

class EditorPropertyTransform3 : public EditorBuiltInProperty {
    GOBCLASS(EditorPropertyTransform3, EditorBuiltInProperty)
public:
    using EditorBuiltInProperty::EditorBuiltInProperty;

    void OnImGuiContent() override;

private:
    std::variant<Isometry3f, Isometry3d, Affine3f, Affine3d, Projective3f, Projective3d> data_;
};

}
