/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-3-24
*/

#pragma once

#include "gobot/core/io/resource.hpp"
#include "gobot/core/rid.h"
#include "gobot/core/math/geometry.hpp"

//#include <dart/dart.hpp>

namespace gobot {

class PhysicsServer3D : public Object {
    GOBCLASS(PhysicsServer3D, Object);

public:
    static PhysicsServer3D *GetInstance();

    enum class ShapeType {
//        Plane,
        Box,
//        Sphere,
//        Cylinder,
//        Capsule,
//        ConvexPolygon,
//        HeightMap,
        Custom,
    };

    RID ShapeCreate(ShapeType shape);

    virtual RID BoxShapeCreate() = 0;

    virtual void ShapeSetData(RID shape, const Variant &data) = 0;

    [[nodiscard]] virtual ShapeType ShapeGetType(RID shape) const = 0;
    [[nodiscard]] virtual Variant ShapeGetData(RID shape) const = 0;

    /***** World API *****/
    virtual RID WorldCreate() = 0;
    // todo: WorldSetParam/WorldGetParam

    /***** Area API *****/

    /* BODY API */
    enum class BodyMode {
        Static,
        Kinematic,
        Rigid,
    };

    virtual RID BodyCreate() = 0;

    // todo: virtual void BodySetMode(RID body, BodyMode mode) = 0;
    // todo: virtual BodyMode BodyGetMode(RID body) = 0;

    // todo: virtual void BodyAddShape(RID body, RID shape, const Affine3 &tfm = Affine3::Identity(), bool disabled = false) = 0;
    // todo: virtual void BodySetShape(RID body, std::size_t shape_idx, RID shape) = 0;
    // todo: virtual void BodySetShapeTransform(RID body, std::size_t shape_idx, const Affine3 &tfm) = 0;

    // todo: virtual std::size_t BodyGetShapeCount(RID body) const = 0;
    // todo: virtual RID BodyGetShape(RID body, std::size_t shape_idx) const = 0;
    // todo: virtual Affine3 BodyGetShapeTransform(RID body, std::size_t shape_idx) const = 0;

    // todo: virtual void BodyRemoveShape(RID body, std::size_t shape_idx) = 0;
    // todo: virtual void BodyClearShapes(RID body) = 0;

    /* COMMON BODY VARS */
//    enum class BodyParameter {
//        Mass,
//        Inertia,
//        CenterOfMass,
//        LinearDamp,
//        AngularDamp,
//        Friction,
//        Restitution,
//    };

//    virtual void BodySetParam(RID body, BodyParameter param, const Variant &value) = 0;
//    virtual Variant BodyGetParam(RID body, BodyParameter param) const = 0;

    /***** MotionParameters *****/
    /***** MotionCollision *****/
    /***** MotionResult *****/

    /* JOINT API */
//    enum class JointType {
//        Revolute,
//        Prismatic,
//        Universal,
//    };

    // todo: virtual RID JointCreate() = 0;

    // todo: virtual void JointClear(RID Joint) = 0;

    // todo: virtual JointType JointGetType(RID joint) = 0;

//    enum class RevoluteJointParam {
//        UpperLimit,
//        LowerLimit,
//        DampingLimit,
//        RestitutionLimit,
//        Damping,
//        Restitution,
//    };

    // todo: JointMakeRevolute
    // todo: Set/GetParam

    /* MISC */
    virtual void Free(RID rid) = 0;

    // todo: virtual void SetActive(bool active) = 0;
    // todo: virtual void Init() = 0;
    // todo: virtual void Step() = 0;
    // todo: virtual void Sync() = 0;
    // todo: virtual void FlushQueries() = 0;
    // todo: virtual void EndSync() = 0;
    // todo: virtual void Finish() = 0;

protected:

private:
    static PhysicsServer3D *singleton_;
};

class PhysicsServer3DManager : public Object {
    GOBCLASS(PhysicsServer3DManager, Object);

public:
    static PhysicsServer3DManager *GetInstance();

    // todo: RegisterServer

    void SetDefaultServer(const String &name, int priority = 0);
    std::size_t FindServerID(const String &name);
    std::size_t GetServersCount();
    String GetServerName(std::size_t id);

    PhysicsServer3D *NewDefaultServer();
    PhysicsServer3D *NewServer(const String &name);

    PhysicsServer3DManager();
    ~PhysicsServer3DManager() override;

protected:

private:
    static PhysicsServer3DManager *singleton_;
};

} // End of namespace gobot