/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-23
*/

#pragma once

#include "gobot/core/object.hpp"
#include "gobot/core/color.hpp"
#include "gobot/core/math/matrix.hpp"
#include "gobot/core/rid.hpp"
#include "render_types.hpp"
#include "rendering_server_globals.hpp"
#include "renderer_viewport.hpp"

namespace gobot {

#define RS RenderServer


class GOBOT_EXPORT RenderServer : public Object {
    GOBCLASS(RenderServer, Object)
public:
    RenderServer();

    ~RenderServer() override;

    RendererType GetRendererType();

    static RenderServer* GetInstance();

    // viewport
    RID ViewportCreate() {
        RID viewport = RSG::viewport->ViewportAllocate();
        RSG::viewport->ViewportInitialize(viewport);
        return viewport;
    }

    void ViewportSetSize(const RID& p_rid, int width, int height) {
        RSG::viewport->ViewportSetSize(p_rid, width, height);
    }

    void* GetRenderTargetColorTextureNativeHandle(const RID& p_view_port);

    // shader
    RID ShaderCreate(ShaderType p_shader_type) {
        RID shader = RSG::material_storage->ShaderAllocate();
        RSG::material_storage->ShaderInitialize(shader, p_shader_type);
        return shader;
    }

    // p_name and p_path is for error debug print.
    void ShaderSetCode(RID p_shader, const String &p_code, const String& p_name = "", const String& p_path = "") {
        RSG::material_storage->ShaderSetCode(p_shader, p_code, p_name, p_path);
    }

    String ShaderGetCode(RID p_shader) {
        return RSG::material_storage->ShaderGetCode(p_shader);
    }

    // shader program
    RID ShaderProgramCreate() {
        RID shader = RSG::material_storage->ShaderProgramAllocate();
        RSG::material_storage->ShaderProgramInitialize(shader);
        return shader;
    }

    void ShaderProgramSetRasterizerShader(RID p_shader_program,
                                          RID p_vs_shader,
                                          RID p_fs_shader,
                                          RID p_geometry_shader = {},
                                          RID p_tess_control_shader = {},
                                          RID p_tess_evaluation_shader = {},
                                          const String& p_name = "") {
        RSG::material_storage->ShaderProgramSetRasterizerShader(p_shader_program,
                                                                p_vs_shader,
                                                                p_fs_shader,
                                                                p_geometry_shader,
                                                                p_tess_control_shader,
                                                                p_tess_evaluation_shader,
                                                                p_name);
    }

    void ShaderProgramSetComputeShader(RID p_shader_program, RID p_comp_shader, const String& p_name = "") {
        RSG::material_storage->ShaderProgramSetComputeShader(p_shader_program,
                                                             p_comp_shader,
                                                             p_name);
    }

    RID ShaderProgramCreate(RID p_compute_shader) {
        RID shader = RSG::material_storage->ShaderProgramAllocate();
        RSG::material_storage->ShaderProgramInitialize(shader);
        return shader;
    }

    RID MaterialCreate() {
        return RID();
    }

    RID MeshCreate();

    void Free(const RID& rid);

    void Draw();

private:
    static RenderServer* s_singleton;

    RendererType renderer_type_;

};

}
