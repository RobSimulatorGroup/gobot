/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-26
*/

#include "gobot/rendering/debug_draw_renderer.hpp"
#include "gobot/rendering/render_types.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/log.hpp"

namespace gobot {

struct DebugVertex
{
    float x;
    float y;
    float z;
    float len;
    uint32_t abgr;

    static void init()
    {
        s_layout
                .begin()
                .add(bgfx::Attrib::Position,  3, bgfx::AttribType::Float)
                .add(bgfx::Attrib::TexCoord0, 1, bgfx::AttribType::Float)
                .add(bgfx::Attrib::Color0,    4, bgfx::AttribType::Uint8, true)
                .end();
    }

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

    static void init()
    {
        s_layout
                .begin()
                .add(bgfx::Attrib::Position,  3, bgfx::AttribType::Float)
                .add(bgfx::Attrib::TexCoord0, 2, bgfx::AttribType::Float)
                .add(bgfx::Attrib::Color0,    4, bgfx::AttribType::Uint8, true)
                .end();
    }

    static bgfx::VertexLayout s_layout;
};

struct DebugShapeVertex
{
    float x;
    float y;
    float z;
    uint8_t indices[4];

    static void init()
    {
        s_layout
                .begin()
                .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
                .add(bgfx::Attrib::Indices,  4, bgfx::AttribType::Uint8)
                .end();
    }

    static bgfx::VertexLayout s_layout;
};

bgfx::VertexLayout DebugShapeVertex::s_layout;

struct DebugMeshVertex
{
    float x;
    float y;
    float z;

    static void init()
    {
        s_layout
                .begin()
                .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
                .end();
    }

    static bgfx::VertexLayout s_layout;
};

static DebugShapeVertex s_quadVertices[4] = {
                {-1.0f, 0.0f,  1.0f, { 0, 0, 0, 0 } },
                { 1.0f, 0.0f,  1.0f, { 0, 0, 0, 0 } },
                {-1.0f, 0.0f, -1.0f, { 0, 0, 0, 0 } },
                { 1.0f, 0.0f, -1.0f, { 0, 0, 0, 0 } },
};

static const uint16_t s_quadIndices[6] = {
                0, 1, 2,
                1, 3, 2,
};

static DebugShapeVertex s_cubeVertices[8] = {
                {-1.0f,  1.0f,  1.0f, { 0, 0, 0, 0 } },
                { 1.0f,  1.0f,  1.0f, { 0, 0, 0, 0 } },
                {-1.0f, -1.0f,  1.0f, { 0, 0, 0, 0 } },
                { 1.0f, -1.0f,  1.0f, { 0, 0, 0, 0 } },
                {-1.0f,  1.0f, -1.0f, { 0, 0, 0, 0 } },
                { 1.0f,  1.0f, -1.0f, { 0, 0, 0, 0 } },
                {-1.0f, -1.0f, -1.0f, { 0, 0, 0, 0 } },
                { 1.0f, -1.0f, -1.0f, { 0, 0, 0, 0 } },
};

static const uint16_t s_cubeIndices[36] = {
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

static const uint8_t s_circleLod[] = {
                37,
                29,
                23,
                17,
                11,
};


DebugDrawRenderer* DebugDrawRenderer::s_singleton = nullptr;

DebugDrawRenderer::DebugDrawRenderer() {
    s_singleton = this;
}

DebugDrawRenderer::~DebugDrawRenderer() {
    s_singleton = nullptr;
}

DebugDrawRenderer* DebugDrawRenderer::GetInstance() {
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize DebugDrawRenderer");
    return s_singleton;
}

}
