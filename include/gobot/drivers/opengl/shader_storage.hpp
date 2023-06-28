/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-27
*/

#pragma once

#include "glad/glad.h"
#include "gobot/core/rid_owner.hpp"
#include "gobot/rendering/shader_storage.hpp"
#include "gobot/scene/resources/shader.hpp"

namespace gobot::opengl {

class GLShaderStorage : public ShaderStorage {
public:
    GLShaderStorage();

    ~GLShaderStorage() override;

    RID ShaderAllocate() override;

    void Initialize(RID shader_rid) override;

    void ShaderSetCode(RID shader_rid, String code) override;

    String ShaderGetCode(RID shader_rid) override;

    void ShaderFree(RID p_rid) override;

    static GLShaderStorage* GetInstance();

private:
    static GLShaderStorage *s_singleton;


    struct Shader {
        GLuint shader_id;
        String code;
        ShaderType shader_type;
    };

    mutable RID_Owner<Shader, true> shader_owner_;
};


}
