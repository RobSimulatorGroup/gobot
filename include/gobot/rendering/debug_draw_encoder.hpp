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

struct DebugVertex
{
    float x;
    float y;
    float z;
    float len;
    uint32_t abgr;

    static void Init();

    static VertexLayout s_layout;
};

struct DebugUvVertex
{
    float x;
    float y;
    float z;
    float u;
    float v;
    uint32_t abgr;

    static void Init();

    static VertexLayout s_layout;
};

struct DebugShapeVertex
{
    float x;
    float y;
    float z;
    uint8_t indices[4];

    static void Init();

    static VertexLayout s_layout;
};

struct DebugMeshVertex
{
    float x;
    float y;
    float z;

    static void Init();

    static VertexLayout s_layout;
};

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

    void SetRenderState(bool depth_test, bool depth_write, bool clock_wise);

    void SetColor(const Color& _abgr);

    void SetLod(uint8_t lod);

    void SetWireframe(bool wire_frame);

    void SetStipple(bool stipple, float scale = 1.0f, float offset = 0.0f);

    void SetSpin(float spin);

    void SetTransform(const Matrix4f& matrix);

    void SetTranslate(float x, float y, float z);

    void PushTransform(const Matrix4f& matrix);

    void PopTransform();

    void MoveTo(float x, float y, float z = 0.0f);

    void MoveTo(const Vector3f& pos);

    void LineTo(float x, float y, float z = 0.0f);

    void LineTo(const Vector3f& pos);

    void Close();

    void Draw(const AABB& aabb);

    void Draw(const Plane& plan);

    void Draw(const Ref<Shape3D>& shape3d);

    void DrawFrustum(const Matrix4f& projection);

    void DrawLineList(const std::vector<Vector3>& vertices,
                      const std::vector<uint16_t>& indices = {});

    void DrawTriList(const std::vector<Vector3>& vertices,
                     const std::vector<uint16_t>& indices = {});

    void DrawArc(Axis axis, float x, float y, float z, float radius, float degrees);

    void DrawCircle(const Vector3& normal, const Vector3& center, float radius, float weight = 0.0f);

    void DrawCircle(Axis axis, float x, float y, float z, float radius, float weight = 0.0f);

    void DrawQuad(const Vector3& normal, const Vector3& center, float size);

    void DrawQuad(TextureHandle handle, const Vector3& normal, const Vector3& _center, float size);

    void DrawCone(const Vector3& from, const Vector3& to, float radius);

    void DrawCylinder(const Vector3& from, const Vector3& to, float radius);

    void DrawCapsule(const Vector3& from, const Vector3& to, float radius);

    void DrawAxis(float x, float y, float z, float len = 1.0f, Axis highlight = Axis::Count, float thickness = 0.0f);

    void DrawGrid(const Vector3& normal, const Vector3& center, uint32_t size = 20, float step = 1.0f);

    void DrawGrid(Axis axis, const Vector3& center, uint32_t size = 20, float step = 1.0f);

    void DrawOrb(float x, float y, float z, float radius, Axis highlight = Axis::Count);

private:
    void Flush();

    void SoftFlush();

private:
    static const uint32_t s_cache_size = 1024;
    static const uint32_t s_stack_size = 16;
    static const uint32_t s_cache_quad_size = 1024;

    DebugVertex  cache_[s_cache_size + 1];
    DebugUvVertex cache_quad_[s_cache_quad_size];
    uint16_t indices_[s_cache_size * 2];

    uint16_t pos_;
    uint16_t pos_quad_;
    uint16_t index_pos_;
    uint16_t vertex_pos_;


    struct MatrixStack
    {
        void reset()
        {
            mtx  = 0;
            num  = 1;
            data = nullptr;
        }

        uint32_t mtx;
        uint16_t num;
        float*   data;
    };

    MatrixStack matrix_stack_[32];
    uint32_t matrix_stack_current_;

    ViewId view_id_;
    uint8_t stack_;
    bool depth_test_less_;

    struct DebugDrawAttrib
    {
        uint64_t state;
        float    offset;
        float    scale;
        float    spin;
        uint32_t abgr;
        bool     stipple;
        bool     wireframe;
        uint8_t  lod;
    };

    DebugDrawAttrib attrib_[s_stack_size];

    enum class DrawState
    {
        None,
        MoveTo,
        LineTo,
        Count
    };


    DrawState draw_state_;

    bgfx::Encoder* encoder_;
    bgfx::Encoder* default_encoder_;
};


}
