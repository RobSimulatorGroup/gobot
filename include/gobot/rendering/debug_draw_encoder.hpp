/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-26
*/

#pragma once

#include "gobot/rendering/render_types.hpp"
#include "gobot/core/color.hpp"
#include "gobot/core/math/matrix.hpp"
#include "gobot/core/math/aabb.hpp"
#include "gobot/core/math/plan.hpp"
#include "gobot/scene/resources/shape_3d.hpp"

namespace gobot {

class DebugDrawEncoder
{
public:
    DebugDrawEncoder();

    ~DebugDrawEncoder();

    void Begin(ViewId view_id, bool depth_test_less = true, bgfx::Encoder* _encoder = nullptr);

    void End();

    void Push();

    void Pop();

    void SetDepthTestLess(bool depth_test_less);

    void SetState(bool depth_test, bool depth_write, bool clock_wise);

    void SetColor(const Color& _abgr);

    void SetLod(uint8_t lod);

    void SetWireframe(bool wire_frame);

    void SetStipple(bool stipple, float scale = 1.0f, float offset = 0.0f);

    void SetSpin(float _spin);

    void SetTransform(const Matrix4f& matrix);

    void SetTranslate(float x, float y, float z);

    void PushTransform(const Matrix4f& matrix);

    void PopTransform();

    void MoveTo(float x, float y, float z = 0.0f);

    void MoveTo(const Vector3f& pos);

    void LineTo(float x, float y, float z = 0.0f);

    void LineTo(const Vector3f& pos);

    void Close();

    // Draw math type
    void Draw(const AABB& aabb);

    void Draw(const Plane& plan);

    void Draw(const Ref<Shape3D>& shape3d);

    void DrawFrustum(const Matrix4f& projection);

//
//    ///
//    void draw(GeometryHandle _handle);
//
//    ///
//    void drawLineList(uint32_t _numVertices, const DdVertex* _vertices, uint32_t _numIndices = 0, const uint16_t* _indices = NULL);
//
//    ///
//    void drawTriList(uint32_t _numVertices, const DdVertex* _vertices, uint32_t _numIndices = 0, const uint16_t* _indices = NULL);
//
//    ///
//    void drawFrustum(const void* _viewProj);
//
//    ///
//    void drawArc(Axis::Enum _axis, float _x, float _y, float _z, float _radius, float _degrees);
//
//    ///
//    void drawCircle(const bx::Vec3& _normal, const bx::Vec3& _center, float _radius, float _weight = 0.0f);
//
//    ///
//    void drawCircle(Axis::Enum _axis, float _x, float _y, float _z, float _radius, float _weight = 0.0f);
//
//    ///
//    void drawQuad(const bx::Vec3& _normal, const bx::Vec3& _center, float _size);
//
//    ///
//    void drawQuad(SpriteHandle _handle, const bx::Vec3& _normal, const bx::Vec3& _center, float _size);
//
//    ///
//    void drawQuad(bgfx::TextureHandle _handle, const bx::Vec3& _normal, const bx::Vec3& _center, float _size);
//
//    ///
//    void drawCone(const bx::Vec3& _from, const bx::Vec3& _to, float _radius);
//
//    ///
//    void drawCylinder(const bx::Vec3& _from, const bx::Vec3& _to, float _radius);
//
//    ///
//    void drawCapsule(const bx::Vec3& _from, const bx::Vec3& _to, float _radius);
//
//    ///
//    void drawAxis(float _x, float _y, float _z, float _len = 1.0f, Axis::Enum _highlight = Axis::Count, float _thickness = 0.0f);
//
//    ///
//    void drawGrid(const bx::Vec3& _normal, const bx::Vec3& _center, uint32_t _size = 20, float _step = 1.0f);
//
//    ///
//    void drawGrid(Axis::Enum _axis, const bx::Vec3& _center, uint32_t _size = 20, float _step = 1.0f);
//
//    ///
//    void drawOrb(float _x, float _y, float _z, float _radius, Axis::Enum _highlight = Axis::Count);
//
//private:
//    static const uint32_t kCacheSize = 1024;
//    static const uint32_t kStackSize = 16;
//    static const uint32_t kCacheQuadSize = 1024;
//    BX_STATIC_ASSERT(kCacheSize >= 3, "Cache must be at least 3 elements.");
//
//    DebugVertex   m_cache[kCacheSize+1];
//    DebugUvVertex m_cacheQuad[kCacheQuadSize];
//    uint16_t m_indices[kCacheSize*2];
//    uint16_t m_pos;
//    uint16_t m_posQuad;
//    uint16_t m_indexPos;
//    uint16_t m_vertexPos;
//    uint32_t m_mtxStackCurrent;
//
//    struct MatrixStack
//    {
//        void reset()
//        {
//            mtx  = 0;
//            num  = 1;
//            data = nullptr;
//        }
//
//        uint32_t mtx;
//        uint16_t num;
//        float*   data;
//    };
//
//    MatrixStack m_mtxStack[32];
//
//    ViewId m_viewId;
//    uint8_t m_stack;
//    bool    m_depthTestLess;
//
//    Attrib m_attrib[kStackSize];
//
//    State::Enum m_state;

    bgfx::Encoder* m_encoder;
    bgfx::Encoder* m_defaultEncoder;
};


}
