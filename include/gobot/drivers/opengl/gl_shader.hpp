/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include "gl_uniform_buffer.hpp"
#include "gobot/graphics/shader.hpp"
#include "gobot/graphics/buffer_layout.hpp"
#include "gl.hpp"


#include <Eigen/Dense>

namespace gobot {

struct GLShaderErrorInfo
{
    GLShaderErrorInfo()
            : shader(0) {};
    uint32_t shader;
    String message[6];
    uint32_t line[6];
};

class GLShader : public Shader
{
private:
    friend class Shader;
    friend class ShaderManager;

public:
    GLShader(const String& filePath);
    GLShader(const uint32_t* vertData, uint32_t vertDataSize, const uint32_t* fragData, uint32_t fragDataSize);
    GLShader(const uint32_t* compData, uint32_t compDataSize);

    ~GLShader();

    void Init();
    void Shutdown() const;
    void Bind() const override;
    void Unbind() const override;

    inline const String& GetName() const override
    {
        return m_Name;
    }
    inline const String& GetFilePath() const override
    {
        return m_Path;
    }

    inline const std::vector<ShaderType> GetShaderTypes() const override
    {
        return m_ShaderTypes;
    }

    static GLuint CompileShader(ShaderType type, String source, uint32_t program, GLShaderErrorInfo& info);
    static uint32_t Compile(std::map<ShaderType, String>* sources, GLShaderErrorInfo& info);

    static void PreProcess(const String& source, std::map<ShaderType, String>* sources);

    static void ReadShaderFile(std::vector<String> lines, std::map<ShaderType, String>* shaders);

    void Parse(std::map<ShaderType, String>* sources);
    void ParseUniform(const String& statement, ShaderType type);
    void ParseUniformStruct(const String& block, ShaderType shaderType);
    void SetUniform(ShaderDataType type, uint8_t* data, uint32_t size, uint32_t offset, const String& name);

    static bool IsTypeStringResource(const String& type);

    int32_t GetUniformLocation(const String& name);
    uint32_t GetHandleInternal() const
    {
        return m_Handle;
    }
    const BufferLayout& GetBufferLayout() const { return m_Layout; }

    void SetUniform1f(const String& name, float value);
    void SetUniform1fv(const String& name, float* value, int32_t count);
    void SetUniform1i(const String& name, int32_t value);
    void SetUniform1ui(const String& name, uint32_t value);
    void SetUniform1iv(const String& name, int32_t* value, int32_t count);
    void SetUniform2f(const String& name, const Eigen::Vector2f& vector);
    void SetUniform3f(const String& name, const Eigen::Vector3f& vector);
    void SetUniform4f(const String& name, const Eigen::Vector4f& vector);
    void SetUniformMat4(const String& name, const Eigen::Matrix4f& matrix);

    void BindUniformBuffer(GLUniformBuffer* buffer, uint32_t slot, const String& name);

    PushConstant* GetPushConstant(uint32_t index) override;

    std::vector<PushConstant>& GetPushConstants() override { return m_PushConstants; }
    void BindPushConstants(CommandBuffer* commandBuffer, Pipeline* pipeline) override;

    DescriptorSetInfo GetDescriptorInfo(uint32_t index) override;

    static void SetUniform1f(uint32_t location, float value);
    static void SetUniform1fv(uint32_t location, float* value, int32_t count);
    static void SetUniform1i(uint32_t location, int32_t value);
    static void SetUniform1ui(uint32_t location, uint32_t value);
    static void SetUniform1iv(uint32_t location, int32_t* value, int32_t count);
    static void SetUniform2f(uint32_t location, const Eigen::Vector2f& vector);
    static void SetUniform3f(uint32_t location, const Eigen::Vector3f& vector);
    static void SetUniform4f(uint32_t location, const Eigen::Vector4f& vector);
    static void SetUniformMat3(uint32_t location, const Eigen::Matrix3f& matrix);
    static void SetUniformMat4(uint32_t location, const Eigen::Matrix4f& matrix);
    static void SetUniformMat4Array(uint32_t location, uint32_t count, const Eigen::Matrix4f& matrix);

    static void MakeDefault();

protected:
    static Shader* CreateFuncGL(const String& filePath);
    static Shader* CreateFromEmbeddedFuncGL(const uint32_t* vertData, uint32_t vertDataSize, const uint32_t* fragData, uint32_t fragDataSize);
    static Shader* CreateCompFromEmbeddedFuncGL(const uint32_t* compData, uint32_t compDataSize);

    void LoadFromData(const uint32_t* data, uint32_t size, ShaderType type, std::map<ShaderType, String>& sources);

private:
    uint32_t m_Handle;
    String m_Name, m_Path;
    String m_Source;

    std::vector<ShaderType> m_ShaderTypes;
    std::map<uint32_t, DescriptorSetInfo> m_DescriptorInfos;

    bool CreateLocations();
    bool SetUniformLocation(const char* szName);

    std::map<uint32_t, uint32_t> m_UniformBlockLocations;
    std::map<uint32_t, uint32_t> m_SampledImageLocations;
    std::map<uint32_t, uint32_t> m_UniformLocations;

//    std::vector<spirv_cross::CompilerGLSL*> m_ShaderCompilers;
    std::vector<PushConstant> m_PushConstants;

    BufferLayout m_Layout;

    void* GetHandle() const override
    {
        return (void*)(size_t)m_Handle;
    }
};

}