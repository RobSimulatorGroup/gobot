/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-26
*/

#include "gobot/rendering/debug_draw_encoder.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

VertexLayout DebugVertex::s_layout;

void DebugVertex::Init() {
    s_layout
        .begin()
        .add(bgfx::Attrib::Position,  3, bgfx::AttribType::Float)
        .add(bgfx::Attrib::TexCoord0, 1, bgfx::AttribType::Float)
        .add(bgfx::Attrib::Color0,    4, bgfx::AttribType::Uint8, true)
        .end();
}


VertexLayout DebugUvVertex::s_layout;

void DebugUvVertex::Init() {
    s_layout
        .begin()
        .add(bgfx::Attrib::Position,  3, bgfx::AttribType::Float)
        .add(bgfx::Attrib::TexCoord0, 2, bgfx::AttribType::Float)
        .add(bgfx::Attrib::Color0,    4, bgfx::AttribType::Uint8, true)
        .end();
}

VertexLayout DebugShapeVertex::s_layout;

void DebugShapeVertex::Init() {
    s_layout
        .begin()
        .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
        .add(bgfx::Attrib::Indices,  4, bgfx::AttribType::Uint8)
        .end();
}

VertexLayout DebugMeshVertex::s_layout;

void DebugMeshVertex::Init() {
    s_layout
        .begin()
        .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
        .end();
}

// define the static data
static DebugShapeVertex s_quad_vertices[4] = {
        {-1.0f, 0.0f,  1.0f, { 0, 0, 0, 0 } },
        { 1.0f, 0.0f,  1.0f, { 0, 0, 0, 0 } },
        {-1.0f, 0.0f, -1.0f, { 0, 0, 0, 0 } },
        { 1.0f, 0.0f, -1.0f, { 0, 0, 0, 0 } },
};

static const uint16_t s_quad_indices[6] = {
        0, 1, 2,
        1, 3, 2,
};

static DebugShapeVertex s_cube_vertices[8] = {
        {-1.0f,  1.0f,  1.0f, { 0, 0, 0, 0 } },
        { 1.0f,  1.0f,  1.0f, { 0, 0, 0, 0 } },
        {-1.0f, -1.0f,  1.0f, { 0, 0, 0, 0 } },
        { 1.0f, -1.0f,  1.0f, { 0, 0, 0, 0 } },
        {-1.0f,  1.0f, -1.0f, { 0, 0, 0, 0 } },
        { 1.0f,  1.0f, -1.0f, { 0, 0, 0, 0 } },
        {-1.0f, -1.0f, -1.0f, { 0, 0, 0, 0 } },
        { 1.0f, -1.0f, -1.0f, { 0, 0, 0, 0 } },
};

static const uint16_t s_cube_indices[36] = {
        0, 1, 2, // 0
        1, 3, 2,
        4, 6, 5, // 2
        5, 6, 7,
        0, 2, 4, // 4
        4, 2, 6,
        1, 5, 3, // 6
        5, 7, 3,
        0, 4, 1, // 8
        4, 5, 1,
        2, 3, 6, // 10
        6, 3, 7,
};

static const uint8_t s_circle_lod[] = {
        37,
        29,
        23,
        17,
        11,
};

static uint8_t GetCircleLod(uint8_t lod)
{
    lod = lod > GetArraySize(s_circle_lod) - 1 ? GetArraySize(s_circle_lod) - 1 : lod;
    return s_circle_lod[lod];
}

static void Circle(float* out, float angle)
{
    float sa = std::sin(angle);
    float ca = std::cos(angle);
    out[0] = sa;
    out[1] = ca;
}

static void Squircle(float* out, float angle)
{
    float sa = std::sin(angle);
    float ca = std::cos(angle);
    out[0] = std::sqrt(std::abs(sa) ) * Sign(sa);
    out[1] = std::sqrt(std::abs(ca) ) * Sign(ca);
}

uint32_t GenSphere(uint8_t sub_div0,
                   void* pos0 = nullptr,
                   uint16_t pos_stride0 = 0,
                   void* normals0 = nullptr,
                   uint16_t normal_stride0 = 0)
{
    if (nullptr != pos0)
    {
        struct Gen
        {
            Gen(void* pos, uint16_t _posStride, void* _normals, uint16_t _normalStride, uint8_t _subdiv)
                    : m_pos( (uint8_t*)pos)
                    , m_normals( (uint8_t*)_normals)
                    , m_posStride(_posStride)
                    , m_normalStride(_normalStride)
            {
                static const float scale = 1.0f;
                static const float golden = 1.6180339887f;
                static const float len = std::sqrt(golden*golden + 1.0f);
                static const float ss = 1.0f/len * scale;
                static const float ll = ss*golden;

                static const Vector3 vv[] =
                        {
                                { -ll, 0.0f, -ss },
                                {  ll, 0.0f, -ss },
                                {  ll, 0.0f,  ss },
                                { -ll, 0.0f,  ss },

                                { -ss,  ll, 0.0f },
                                {  ss,  ll, 0.0f },
                                {  ss, -ll, 0.0f },
                                { -ss, -ll, 0.0f },

                                { 0.0f, -ss,  ll },
                                { 0.0f,  ss,  ll },
                                { 0.0f,  ss, -ll },
                                { 0.0f, -ss, -ll },
                        };

                m_numVertices = 0;

                triangle(vv[ 0], vv[ 4], vv[ 3], scale, _subdiv);
                triangle(vv[ 0], vv[10], vv[ 4], scale, _subdiv);
                triangle(vv[ 4], vv[10], vv[ 5], scale, _subdiv);
                triangle(vv[ 5], vv[10], vv[ 1], scale, _subdiv);
                triangle(vv[ 5], vv[ 1], vv[ 2], scale, _subdiv);
                triangle(vv[ 5], vv[ 2], vv[ 9], scale, _subdiv);
                triangle(vv[ 5], vv[ 9], vv[ 4], scale, _subdiv);
                triangle(vv[ 3], vv[ 4], vv[ 9], scale, _subdiv);

                triangle(vv[ 0], vv[ 3], vv[ 7], scale, _subdiv);
                triangle(vv[ 0], vv[ 7], vv[11], scale, _subdiv);
                triangle(vv[11], vv[ 7], vv[ 6], scale, _subdiv);
                triangle(vv[11], vv[ 6], vv[ 1], scale, _subdiv);
                triangle(vv[ 1], vv[ 6], vv[ 2], scale, _subdiv);
                triangle(vv[ 2], vv[ 6], vv[ 8], scale, _subdiv);
                triangle(vv[ 8], vv[ 6], vv[ 7], scale, _subdiv);
                triangle(vv[ 8], vv[ 7], vv[ 3], scale, _subdiv);

                triangle(vv[ 0], vv[11], vv[10], scale, _subdiv);
                triangle(vv[ 1], vv[10], vv[11], scale, _subdiv);
                triangle(vv[ 2], vv[ 8], vv[ 9], scale, _subdiv);
                triangle(vv[ 3], vv[ 9], vv[ 8], scale, _subdiv);
            }

            void addVert(const Vector3& _v)
            {
//                bx::store(m_pos, _v);
//                m_pos += m_posStride;
//
//                if (NULL != m_normals)
//                {
//                    const Vector3 normal = _v.normalized();
//                    bx::store(m_normals, normal);
//
//                    m_normals += m_normalStride;
//                }
//
//                m_numVertices++;
            }

            void triangle(const Vector3& v0, const Vector3& v1, const Vector3& v2, float _scale, uint8_t subdiv)
            {
                if (0 == subdiv)
                {
                    addVert(v0);
                    addVert(v1);
                    addVert(v2);
                }
                else
                {
//                    const Vector3 v01 = bx::mul(bx::normalize(bx::add(_v0, _v1) ), _scale);
//                    const Vector3 v12 = bx::mul(bx::normalize(bx::add(_v1, _v2) ), _scale);
//                    const Vector3 v20 = bx::mul(bx::normalize(bx::add(_v2, _v0) ), _scale);
//
//                    --subdiv;
//                    triangle(v0, v01, v20, _scale, subdiv);
//                    triangle(v1, v12, v01, _scale, subdiv);
//                    triangle(v2, v20, v12, _scale, subdiv);
//                    triangle(v01, v12, v20, _scale, subdiv);
                }
            }

            uint8_t* m_pos;
            uint8_t* m_normals;
            uint16_t m_posStride;
            uint16_t m_normalStride;
            uint32_t m_numVertices;

        } gen(pos0, pos_stride0, normals0, normal_stride0, sub_div0);
    }

    uint32_t numVertices = 20 * 3* std::max((uint32_t)1, (uint32_t)std::pow(4.0f, sub_div0) );
    return numVertices;
}

//static const bgfx::EmbeddedShader s_embeddedShaders[] = {
//    BGFX_EMBEDDED_SHADER(vs_debugdraw_lines),
//    BGFX_EMBEDDED_SHADER(fs_debugdraw_lines),
//    BGFX_EMBEDDED_SHADER(vs_debugdraw_lines_stipple),
//    BGFX_EMBEDDED_SHADER(fs_debugdraw_lines_stipple),
//    BGFX_EMBEDDED_SHADER(vs_debugdraw_fill),
//    BGFX_EMBEDDED_SHADER(vs_debugdraw_fill_mesh),
//    BGFX_EMBEDDED_SHADER(fs_debugdraw_fill),
//    BGFX_EMBEDDED_SHADER(vs_debugdraw_fill_lit),
//    BGFX_EMBEDDED_SHADER(vs_debugdraw_fill_lit_mesh),
//    BGFX_EMBEDDED_SHADER(fs_debugdraw_fill_lit),
//    BGFX_EMBEDDED_SHADER(vs_debugdraw_fill_texture),
//    BGFX_EMBEDDED_SHADER(fs_debugdraw_fill_texture),
//
//    BGFX_EMBEDDED_SHADER_END()
//};

#define SPRITE_TEXTURE_SIZE 1024


enum class DebugDrawProgram
{
    Lines,
    LinesStipple,
    Fill,
    FillMesh,
    FillLit,
    FillLitMesh,
    FillTexture,

    Count
};

struct DebugMesh
{
    enum Enum
    {
        Sphere0,
        Sphere1,
        Sphere2,
        Sphere3,

        Cone0,
        Cone1,
        Cone2,
        Cone3,

        Cylinder0,
        Cylinder1,
        Cylinder2,
        Cylinder3,

        Capsule0,
        Capsule1,
        Capsule2,
        Capsule3,

        Quad,

        Cube,

        Count,

        SphereMaxLod   = Sphere3   - Sphere0,
        ConeMaxLod     = Cone3     - Cone0,
        CylinderMaxLod = Cylinder3 - Cylinder0,
        CapsuleMaxLod  = Capsule3  - Capsule0,
    };

    uint32_t start_vertex;
    uint32_t num_vertices;
    uint32_t start_index[2];
    uint32_t num_indices[2];
};


DebugDrawEncoder::DebugDrawEncoder() {

}

DebugDrawEncoder::~DebugDrawEncoder() {

}

void DebugDrawEncoder::Flush() {

}

void DebugDrawEncoder::SoftFlush() {
//    if (pos_ == uint16_t(BX_COUNTOF(m_cache))) {
//        Flush();
//    }
}

void DebugDrawEncoder::MoveTo(float x, float y, float z)
{
    CRASH_COND_MSG(draw_state_ == DrawState::Count, "");

    SoftFlush();

//    state_ = State::MoveTo;
//
//    DebugVertex& vertex = m_cache[m_pos];
//    vertex.m_x = _x;
//    vertex.m_y = _y;
//    vertex.m_z = _z;
//
//    Attrib& attrib = m_attrib[m_stack];
//    vertex.m_abgr = attrib.m_abgr;
//    vertex.m_len  = attrib.m_offset;
//
//    m_vertexPos = m_pos;
}

void DebugDrawEncoder::MoveTo(const Vector3& _pos)
{
//    BX_ASSERT(State::Count != m_state, "");
//    moveTo(_pos.x, _pos.y, _pos.z);
}

//void DebugDrawEncoder::MoveTo(Axis _axis, float _x, float _y)
//{
////    moveTo(getPoint(_axis, _x, _y) );
//}

void DebugDrawEncoder::DrawGrid(const Vector3& normal, const Vector3& center, uint32_t size, float step) {
    const auto& attrib = attrib_[stack_];

    Vector3 udir;
    Vector3 vdir;
//    bx::calcTangentFrame(udir, vdir, normal, attrib.spin);

    udir = udir * step;
    vdir = vdir * step;

    const uint32_t num = (size / 2) * 2 + 1;
    const float half_extent = size/ 2.0;

    const Vector3 umin = udir * (-half_extent);
    const Vector3 umax = udir * ( half_extent);
    const Vector3 vmin = vdir * (-half_extent);
    const Vector3 vmax = vdir * ( half_extent);

    Vector3 xs = center + (umin + vmin);
    Vector3 xe = center + (umax + vmin);
    Vector3 ys = center + (umin + vmin);
    Vector3 ye = center + (umin + vmax);

    for (uint32_t ii = 0; ii < num; ++ii)
    {
        MoveTo(xs);
//        LineTo(xe);
        xs = xs + vdir;
        xe = xe + vdir;

        MoveTo(ys);
//        LineTo(ye);
        ys = ys + udir;
        ye = ye + udir;
    }
}


}
