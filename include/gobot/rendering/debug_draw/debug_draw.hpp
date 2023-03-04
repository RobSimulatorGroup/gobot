/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-27
*/

#pragma once

#include "gobot/rendering/render_types.hpp"
#include "gobot/scene/resources/shape_3d.hpp"
#include "gobot/core/math/matrix.hpp"
#include <bx/allocator.h>
#include <bx/bounds.h>
#include <bgfx/bgfx.h>

namespace gobot {

struct DdVertex
{
    float x, y, z;
};

struct SpriteHandle { uint16_t idx; };
inline bool isValid(SpriteHandle _handle) { return _handle.idx != UINT16_MAX; }

struct GeometryHandle { uint16_t idx; };
inline bool isValid(GeometryHandle _handle) { return _handle.idx != UINT16_MAX; }

void ddInit(bx::AllocatorI* _allocator = nullptr);

void ddShutdown();

SpriteHandle ddCreateSprite(uint16_t _width, uint16_t _height, const void* _data);

void ddDestroy(SpriteHandle _handle);

GeometryHandle ddCreateGeometry(uint32_t _numVertices, const DdVertex* _vertices, uint32_t _numIndices = 0, const void* _indices = nullptr, bool _index32 = false);

void ddDestroy(GeometryHandle _handle);

struct DebugDrawEncoder
{
    DebugDrawEncoder();

    ~DebugDrawEncoder();

    void Begin(uint16_t viewId, bool depth_test_less = true, bgfx::Encoder* encoder = nullptr);

    void End();

    void Push();

    void Pop();

    void SetDepthTestLess(bool depth_test_less);

    void SetState(bool depth_test, bool depth_write, bool clockwise);

    void SetColor(uint32_t abgr);

    void SetLod(uint8_t lod);

    void SetWireframe(bool wireframe);

    void SetStipple(bool stipple, float scale = 1.0f, float offset = 0.0f);

    void SetSpin(float spin);

    void SetTransform(const void* mtx);

    void SetTranslate(float x, float y, float z);

    void PushTransform(const void* mtx);

    void PopTransform();

    void MoveTo(float _x, float _y, float _z = 0.0f);

    void MoveTo(const Vector3f & _pos);

    void LineTo(float _x, float _y, float _z = 0.0f);

    void LineTo(const Vector3f& _pos);

    void Close();

    void DrawShape3D(const Ref<Shape3D>& shape_3d);

    void Draw(const bx::Aabb& _aabb);

    void Draw(const bx::Cylinder& _cylinder);

    void Draw(const bx::Capsule& _capsule);

    void Draw(const bx::Disk& _disk);

    void Draw(const bx::Obb& _obb);

    void Draw(const bx::Sphere& _sphere);

    void Draw(const bx::Triangle& _triangle);

    void Draw(const bx::Cone& _cone);

    void Draw(GeometryHandle _handle);

    void DrawLineList(uint32_t _numVertices, const DdVertex* _vertices, uint32_t _numIndices = 0, const uint16_t* _indices = nullptr);

    void DrawTriList(uint32_t _numVertices, const DdVertex* _vertices, uint32_t _numIndices = 0, const uint16_t* _indices = nullptr);

    void DrawFrustum(const void* _viewProj);

    void DrawArc(Axis _axis, float _x, float _y, float _z, float _radius, float _degrees);

    void DrawCircle(const bx::Vec3& _normal, const bx::Vec3& _center, float _radius, float _weight = 0.0f);

    void DrawCircle(Axis _axis, float _x, float _y, float _z, float _radius, float _weight = 0.0f);

    void DrawQuad(const bx::Vec3& _normal, const bx::Vec3& _center, float _size);

    void DrawQuad(SpriteHandle _handle, const bx::Vec3& _normal, const bx::Vec3& _center, float _size);

    void DrawQuad(bgfx::TextureHandle _handle, const bx::Vec3& _normal, const bx::Vec3& _center, float _size);

    void DrawCone(const bx::Vec3& _from, const bx::Vec3& _to, float _radius);

    void DrawCylinder(const bx::Vec3& _from, const bx::Vec3& _to, float _radius);

    void DrawCapsule(const bx::Vec3& _from, const bx::Vec3& _to, float _radius);

    void DrawWorldAxis(float len = 1.0f, Axis highlight = Axis::Count);

    void DrawAxis(float _x, float _y, float _z, float _len = 1.0f, Axis _highlight = Axis::Count, float _thickness = 0.0f);

    void DrawGrid(const bx::Vec3& _normal, const bx::Vec3& _center, uint32_t _size = 20, float _step = 1.0f);

    void DrawGrid(Axis _axis, const bx::Vec3& _center, uint32_t _size = 20, float _step = 1.0f);

    void DrawOrb(float _x, float _y, float _z, float _radius, Axis _highlight = Axis::Count);

    BX_ALIGN_DECL_CACHE_LINE(uint8_t) m_internal[50<<10];
};

class DebugDrawEncoderScopePush
{
public:
    DebugDrawEncoderScopePush(DebugDrawEncoder& _dde);

    ~DebugDrawEncoderScopePush();

private:
    DebugDrawEncoder& m_dde;
};

}
