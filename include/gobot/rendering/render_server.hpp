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
        RID shader = RSG::shader_storage->ShaderAllocate();
        RSG::shader_storage->ShaderInitialize(shader, p_shader_type);
        return shader;
    }

    void ShaderSetCode(RID p_shader, const String &p_code) {
        RSG::shader_storage->ShaderSetCode(p_shader, p_code);
    }

    String ShaderGetCode(RID p_shader) {
        return RSG::shader_storage->ShaderGetCode(p_shader);
    }

    // shader program
    RID ShaderProgramCreate(const Ref<Shader>& p_vs_shader,
                            const Ref<Shader>& p_fs_shader,
                            const Ref<Shader>& p_geometry_shader = nullptr,
                            const Ref<Shader>& p_tess_control_shader = nullptr,
                            const Ref<Shader>& p_tess_evaluation_shader = nullptr) {
        RID shader = RSG::shader_program_storage->ShaderProgramAllocate();
        RSG::shader_program_storage->ShaderProgramInitialize(shader,
                                                             p_vs_shader,
                                                             p_fs_shader,
                                                             p_geometry_shader,
                                                             p_tess_control_shader,
                                                             p_tess_evaluation_shader);
        return shader;
    }

    RID ShaderProgramCreate(const Ref<Shader>& p_compute_shader) {
        RID shader = RSG::shader_program_storage->ShaderProgramAllocate();
        RSG::shader_program_storage->ShaderProgramInitialize(shader, p_compute_shader);
        return shader;
    }

    void Free(const RID& rid);

    RID CreateMesh();

    void Draw();

private:
    static RenderServer* s_singleton;

    RendererType renderer_type_;

};

}
