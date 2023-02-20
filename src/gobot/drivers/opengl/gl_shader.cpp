/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#include "gobot/drivers/opengl/gl_shader.hpp"
#include "gobot/log.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

bool IGNORE_LINES        = false;
static ShaderType s_Type = ShaderType::UNKNOWN;

uint32_t GetStrideFromOpenGLFormat(uint32_t format)
{
    // switch(format)
    //{
    //     //                case VK_FORMAT_R8_SINT:
    //     //                return sizeof(int);
    //     //                case VK_FORMAT_R32_SFLOAT:
    //     //                return sizeof(float);
    //     //                case VK_FORMAT_R32G32_SFLOAT:
    //     //                return sizeof(glm::vec2);
    //     //                case VK_FORMAT_R32G32B32_SFLOAT:
    //     //                return sizeof(glm::vec3);
    //     //                case VK_FORMAT_R32G32B32A32_SFLOAT:
    //     //                return sizeof(glm::vec4);
    // default:
    //     //LUMOS_LOG_ERROR("Unsupported Format {0}", format);
    //     return 0;
    // }

    return 0;
}


GLShader::GLShader(const String & filePath)
{
//    m_Name = StringUtilities::GetFileName(filePath);
//    m_Path = StringUtilities::GetFileLocation(filePath);
//
//    m_Source = VFS::Get().ReadTextFile(filePath);

    Init();
}

GLShader::GLShader(const uint32_t* vertData, uint32_t vertDataSize, const uint32_t* fragData, uint32_t fragDataSize)
{
    std::map<ShaderType, String>* sources = new std::map<ShaderType, String>();

    LoadFromData(vertData, vertDataSize, ShaderType::VERTEX, *sources);
    LoadFromData(fragData, fragDataSize, ShaderType::FRAGMENT, *sources);

    for(auto& source : *sources)
    {
        m_ShaderTypes.push_back(source.first);
    }

    GLShaderErrorInfo error;
    m_Handle = Compile(sources, error);

    if(!m_Handle)
    {
        LOG_ERROR("{0} - {1}", error.message[error.shader], m_Name);
    }
    else
    {
        LOG_INFO("Successfully compiled shader: {0}", m_Name);
    }

    CreateLocations();

    delete sources;
}

GLShader::GLShader(const uint32_t* compData, uint32_t compDataSize)
{
    std::map<ShaderType, String>* sources = new std::map<ShaderType, String>();

    LoadFromData(compData, compDataSize, ShaderType::COMPUTE, *sources);

    for(auto& source : *sources)
    {
        m_ShaderTypes.push_back(source.first);
    }

    GLShaderErrorInfo error;
    m_Handle = Compile(sources, error);

    if(!m_Handle)
    {
        LOG_ERROR("{0} - {1}", error.message[error.shader], m_Name);
    }
    else
    {
        LOG_INFO("Successfully compiled shader: {0}", m_Name);
    }

    CreateLocations();

    delete sources;
}

GLShader::~GLShader()
{
    Shutdown();

    for(auto& pc : m_PushConstants)
        delete[] pc.data;
}

void GLShader::Init()
{
    std::map<ShaderType, String>* sources = new std::map<ShaderType, String>();
    PreProcess(m_Source, sources);

    for(auto& file : *sources)
    {
//        auto fileSize    = FileSystem::GetFileSize(m_Path + file.second); // TODO: once process
//        uint32_t* source = reinterpret_cast<uint32_t*>(FileSystem::ReadFile(m_Path + file.second));
//        LoadFromData(source, uint32_t(fileSize), file.first, *sources);
    }

    for(auto& source : *sources)
    {
        m_ShaderTypes.push_back(source.first);
    }

    GLShaderErrorInfo error;
    m_Handle = Compile(sources, error);

    if(!m_Handle)
    {
        LOG_ERROR("{0} - {1}", error.message[error.shader], m_Name);
    }
    else
    {
        LOG_INFO("Successfully compiled shader: {0}", m_Name);
    }

    CreateLocations();

    delete sources;
}

void GLShader::Shutdown() const
{
    glDeleteProgram(m_Handle);
}

GLuint HashValue(const char* szString)
{
    const char* c      = szString;
    GLuint dwHashValue = 0x00000000;

    while(*c)
    {
        dwHashValue = (dwHashValue << 5) - dwHashValue + (*c == '/' ? '\\' : *c);
        c++;
    }

    return dwHashValue ? dwHashValue : 0xffffffff;
}

void GLShader::SetUniform(ShaderDataType type, uint8_t* data, uint32_t size, uint32_t offset, const String& name)
{

    GLuint hashName = HashValue(name.toStdString().c_str());

    if(m_UniformLocations.find(hashName) == m_UniformLocations.end())
    {
        GLuint location = glGetUniformLocation(m_Handle, name.toStdString().c_str());

        if(location != GL_INVALID_INDEX)
        {
            m_UniformLocations[hashName] = location;
        }
        else
        {
            LOG_WARN("Invalid uniform location {0}", name);
        }
    }

    auto location = m_UniformLocations[hashName];
    if(location == -1)
    {
        LOG_ERROR("Couldnt Find Uniform In Shader: {0}", name);
        return;
    }

    switch(type)
    {
        case ShaderDataType::FLOAT32:
            SetUniform1f(location, *reinterpret_cast<float*>(&data[offset]));
            break;
        case ShaderDataType::INT32:
            SetUniform1i(location, *reinterpret_cast<int32_t*>(&data[offset]));
            break;
        case ShaderDataType::UINT:
            SetUniform1ui(location, *reinterpret_cast<uint32_t*>(&data[offset]));
            break;
        case ShaderDataType::INT:
            SetUniform1i(location, *reinterpret_cast<int*>(&data[offset]));
            break;
        case ShaderDataType::VEC2:
            SetUniform2f(location, *reinterpret_cast<Eigen::Vector2f*>(&data[offset]));
            break;
        case ShaderDataType::VEC3:
            SetUniform3f(location, *reinterpret_cast<Eigen::Vector3f*>(&data[offset]));
            break;
        case ShaderDataType::VEC4:
            SetUniform4f(location, *reinterpret_cast<Eigen::Vector4f*>(&data[offset]));
            break;
        case ShaderDataType::MAT3:
            SetUniformMat3(location, *reinterpret_cast<Eigen::Matrix3f*>(&data[offset]));
            break;
        case ShaderDataType::MAT4:
            SetUniformMat4(location, *reinterpret_cast<Eigen::Matrix4f*>(&data[offset]));
            break;
        default:
            CRASH_COND_MSG(false, "Unknown type!");
    }
}

void GLShader::BindPushConstants(CommandBuffer* commandBuffer, Pipeline* pipeline)
{
    int index = 0;
    for(auto pc : m_PushConstants)
    {
        for(auto& member : pc.m_Members)
        {
            SetUniform(member.type, pc.data, member.size, member.offset, member.fullName);
        }
    }
}

DescriptorSetInfo GLShader::GetDescriptorInfo(uint32_t index) {
    if(m_DescriptorInfos.find(index) != m_DescriptorInfos.end())
    {
        return m_DescriptorInfos[index];
    }

    LOG_WARN("DescriptorDesc not found. Index = {0}", index);
    return DescriptorSetInfo();
}

bool GLShader::CreateLocations()
{
//    for(auto& compiler : m_ShaderCompilers)
//    {
//        const spirv_cross::ShaderResources shaderResources = compiler->get_shader_resources();
//
//        for(const auto& itUniform : shaderResources.uniform_buffers)
//        {
//            if(compiler->get_type(itUniform.type_id).basetype == spirv_cross::SPIRType::Struct)
//            {
//                SetUniformLocation(itUniform.name.c_str());
//            }
//        }
//
//        for(const auto& itUniform : shaderResources.push_constant_buffers)
//        {
//            if(compiler->get_type(itUniform.type_id).basetype == spirv_cross::SPIRType::Struct)
//            {
//                auto name = compiler->get_name(itUniform.base_type_id);
//                SetUniformLocation(name.c_str());
//            }
//        }
//    }
    return true;
}

void GLShader::BindUniformBuffer(GLUniformBuffer* buffer, uint32_t slot, const String& name)
{
    GLuint nameInt         = HashValue(name.toStdString().c_str());
    const auto& itLocation = m_UniformBlockLocations.find(nameInt);
    glUniformBlockBinding(m_Handle, itLocation->second, slot);
}

bool GLShader::SetUniformLocation(const char* szName)
{
    GLuint name = HashValue(szName);

    if(m_UniformBlockLocations.find(name) == m_UniformBlockLocations.end())
    {
        GLuint location = glGetUniformBlockIndex(m_Handle, szName);

        if(location != GL_INVALID_INDEX)
        {
            m_UniformBlockLocations[name] = location;
            glUniformBlockBinding(m_Handle, location, location);
            return true;
        }
    }

    return false;
}

PushConstant* GLShader::GetPushConstant(uint32_t index) {
    CRASH_COND_MSG(index < m_PushConstants.size(), "Push constants out of bounds");
    return &m_PushConstants[index];
}

void GLShader::PreProcess(const String& source, std::map<ShaderType, String>* sources)
{
    s_Type                         = ShaderType::UNKNOWN;
//    std::vector<String> lines = StringUtilities::GetLines(source);
//    ReadShaderFile(lines, sources);
}

void GLShader::ReadShaderFile(std::vector<String> lines, std::map<ShaderType, String>* shaders)
{
//    for(uint32_t i = 0; i < lines.size(); i++)
//    {
//        String str = String(lines[i]);
//        str             = StringUtilities::StringReplace(str, '\t');
//
//        if(IGNORE_LINES)
//        {
//            if(StringUtilities::StartsWith(str, "#end"))
//            {
//                IGNORE_LINES = false;
//            }
//        }
//        else if(StringUtilities::StartsWith(str, "#shader"))
//        {
//            if(StringUtilities::StringContains(str, "vertex"))
//            {
//                s_Type                                         = ShaderType::VERTEX;
//                std::map<ShaderType, String>::iterator it = shaders->begin();
//                shaders->insert(it, std::pair<ShaderType, String>(s_Type, ""));
//            }
//            else if(StringUtilities::StringContains(str, "geometry"))
//            {
//                s_Type                                         = ShaderType::GEOMETRY;
//                std::map<ShaderType, String>::iterator it = shaders->begin();
//                shaders->insert(it, std::pair<ShaderType, String>(s_Type, ""));
//            }
//            else if(StringUtilities::StringContains(str, "fragment"))
//            {
//                s_Type                                         = ShaderType::FRAGMENT;
//                std::map<ShaderType, String>::iterator it = shaders->begin();
//                shaders->insert(it, std::pair<ShaderType, String>(s_Type, ""));
//            }
//            else if(StringUtilities::StringContains(str, "tess_cont"))
//            {
//                s_Type                                         = ShaderType::TESSELLATION_CONTROL;
//                std::map<ShaderType, String>::iterator it = shaders->begin();
//                shaders->insert(it, std::pair<ShaderType, String>(s_Type, ""));
//            }
//            else if(StringUtilities::StringContains(str, "tess_eval"))
//            {
//                s_Type                                         = ShaderType::TESSELLATION_EVALUATION;
//                std::map<ShaderType, String>::iterator it = shaders->begin();
//                shaders->insert(it, std::pair<ShaderType, String>(s_Type, ""));
//            }
//            else if(StringUtilities::StringContains(str, "compute"))
//            {
//                s_Type                                         = ShaderType::COMPUTE;
//                std::map<ShaderType, String>::iterator it = shaders->begin();
//                shaders->insert(it, std::pair<ShaderType, String>(s_Type, ""));
//            }
//            else if(StringUtilities::StringContains(str, "end"))
//            {
//                s_Type = ShaderType::UNKNOWN;
//            }
//        }
//        else if(StringUtilities::StartsWith(str, "#include"))
//        {
//            String rem  = "#include ";
//            String file = String(str);
//            if(strstr(file.c_str(), rem.c_str()))
//            {
//                String::size_type j = file.find(rem);
//                if(j != String::npos)
//                    file.erase(j, rem.length());
//                file = StringUtilities::StringReplace(file, '\"');
//                LUMOS_LOG_WARN("Including file \'{0}\' into shader.", file);
//                VFS::Get().ReadTextFile(file);
//                ReadShaderFile(StringUtilities::GetLines(VFS::Get().ReadTextFile(file)), shaders);
//            }
//        }
//        else if(StringUtilities::StartsWith(str, "#if"))
//        {
//            String rem = "#if ";
//            String def = String(str);
//            if(strstr(def.c_str(), rem.c_str()))
//            {
//                String::size_type j = def.find(rem);
//                if(j != String::npos)
//                    def.erase(j, rem.length());
//                def = StringUtilities::StringReplace(def, '\"');
//
//                if(def == "0")
//                {
//                    IGNORE_LINES = true;
//                }
//            }
//        }
//        else if(s_Type != ShaderType::UNKNOWN)
//        {
//            shaders->at(s_Type).append(lines[i]);
//            /// Shaders->at(s_Type).append("\n");
//        }
//    }
}

uint32_t GLShader::Compile(std::map<ShaderType, String>* sources, GLShaderErrorInfo& info)
{
    uint32_t program = glCreateProgram();

    std::vector<GLuint> shaders;

    String glVersion;

#ifndef LUMOS_PLATFORM_MOBILE
    glVersion = "#version 410 core \n";
#else
    glVersion = "#version 300 es \n precision highp float; \n precision highp int; \n";
#endif

    for(auto source : *sources)
    {
        // source.second.insert(0, glVersion);
        // LUMOS_LOG_INFO(source.second);
        shaders.push_back(CompileShader(source.first, source.second, program, info));
    }

    for(unsigned int shader : shaders)
        glAttachShader(program, shader);

    glLinkProgram(program);

    GLint result;
    glGetProgramiv(program, GL_LINK_STATUS, &result);
    if(result == GL_FALSE)
    {
        GLint length;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        std::vector<char> error(length);
        glGetProgramInfoLog(program, length, &length, error.data());
        String errorMessage(error.data());
        int32_t lineNumber = -1;
        sscanf(error.data(), "%*s %*d:%d", &lineNumber);
        info.shader = 3;
        info.message[info.shader] += "Failed to link shader!\n";
        info.line[info.shader] = 0;
        info.message[info.shader] += errorMessage;

        LOG_ERROR("{}", info.message[info.shader]);
        return 0;
    }

    glValidateProgram(program);

    for(int z = 0; z < shaders.size(); z++)
        glDetachShader(program, shaders[z]);

    for(int z = 0; z < shaders.size(); z++)
        glDeleteShader(shaders[z]);

    return program;
}

GLenum TypeToGL(ShaderType type)
{
    switch(type)
    {
        case ShaderType::VERTEX:
            return GL_VERTEX_SHADER;
        case ShaderType::FRAGMENT:
            return GL_FRAGMENT_SHADER;
#ifndef LUMOS_PLATFORM_MOBILE
        case ShaderType::GEOMETRY:
            return GL_GEOMETRY_SHADER;
        case ShaderType::TESSELLATION_CONTROL:
            return GL_TESS_CONTROL_SHADER;
        case ShaderType::TESSELLATION_EVALUATION:
            return GL_TESS_EVALUATION_SHADER;
        case ShaderType::COMPUTE:
            return GL_COMPUTE_SHADER;
#endif
        default:
            LOG_ERROR("Unsupported Shader Type");
            return GL_VERTEX_SHADER;
    }
}

String TypeToString(ShaderType type)
{
    switch(type)
    {
        case ShaderType::VERTEX:
            return "GL_VERTEX_SHADER";
        case ShaderType::FRAGMENT:
            return "GL_FRAGMENT_SHADER";
        case ShaderType::GEOMETRY:
            return "GL_GEOMETRY_SHADER";
        case ShaderType::TESSELLATION_CONTROL:
            return "GL_TESS_CONTROL_SHADER";
        case ShaderType::TESSELLATION_EVALUATION:
            return "GL_TESS_EVALUATION_SHADER";
        case ShaderType::COMPUTE:
            return "GL_COMPUTE_SHADER";
        case ShaderType::UNKNOWN:
            return "UNKOWN_SHADER";
    }
    return "N/A";
}

GLuint GLShader::CompileShader(ShaderType type, String source, uint32_t program, GLShaderErrorInfo& info)
{
    const char* cstr = source.toStdString().c_str();

    GLuint shader = glCreateShader(TypeToGL(type));
    glShaderSource(shader, 1, &cstr, NULL);
    glCompileShader(shader);

    GLint result;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
    if(result == GL_FALSE)
    {
        GLint length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::vector<char> error(length);
        glGetShaderInfoLog(shader, length, &length, error.data());
        String errorMessage(error.data());
        int32_t lineNumber;
        sscanf(error.data(), "%*s %*d:%d", &lineNumber);
        info.shader = static_cast<uint32_t>(type);
        info.message[info.shader] += "Failed to compile " + TypeToString(type) + " shader!\n";

        info.line[info.shader] = lineNumber;
        info.message[info.shader] += errorMessage;
        glDeleteShader(shader);

        LOG_ERROR("{}", info.message[info.shader]);
        return 0;
    }
    return shader;
}

void GLShader::Bind() const
{
    if(s_CurrentlyBound != this)
    {
        glUseProgram(m_Handle);
        s_CurrentlyBound = this;
    }
}

void GLShader::Unbind() const
{
    glUseProgram(0);
    s_CurrentlyBound = nullptr;
}

bool GLShader::IsTypeStringResource(const String& type)
{
    if(type == "sampler2D")
        return true;
    if(type == "samplerCube")
        return true;
    if(type == "sampler2DShadow")
        return true;
    if(type == "sampler2DArrayShadow")
        return true;
    return false;
}

GLint GLShader::GetUniformLocation(const String& name)
{
    GLuint hashName = HashValue(name.toStdString().c_str());

    if(m_UniformLocations.find(hashName) == m_UniformLocations.end())
    {
        GLuint location = glGetUniformLocation(m_Handle, name.toStdString().c_str());

        if(location != GL_INVALID_INDEX)
        {
            m_UniformLocations[hashName] = location;
        }
        else
        {
            LOG_WARN("Invalid uniform location {0}", name);
        }
    }

    auto location = m_UniformLocations[hashName];
    if(location == -1)
    {
        LOG_ERROR("Couldnt Find Uniform In Shader: {0}", name);
        return location;
    }

    return location;
}

void GLShader::SetUniform1f(const String& name, float value)
{
    SetUniform1f(GetUniformLocation(name), value);
}

void GLShader::SetUniform1fv(const String& name, float* value, int32_t count)
{
    SetUniform1fv(GetUniformLocation(name), value, count);
}

void GLShader::SetUniform1i(const String& name, int32_t value)
{
    SetUniform1i(GetUniformLocation(name), value);
}

void GLShader::SetUniform1ui(const String& name, uint32_t value)
{
    SetUniform1ui(GetUniformLocation(name), value);
}

void GLShader::SetUniform1iv(const String& name, int32_t* value, int32_t count)
{
    SetUniform1iv(GetUniformLocation(name), value, count);
}

void GLShader::SetUniform2f(const String& name, const Eigen::Vector2f& vector)
{
    SetUniform2f(GetUniformLocation(name), vector);
}

void GLShader::SetUniform3f(const String& name, const Eigen::Vector3f& vector)
{
    SetUniform3f(GetUniformLocation(name), vector);
}

void GLShader::SetUniform4f(const String& name, const Eigen::Vector4f& vector)
{
    SetUniform4f(GetUniformLocation(name), vector);
}

void GLShader::SetUniformMat4(const String& name, const Eigen::Matrix4f& matrix)
{
    SetUniformMat4(GetUniformLocation(name), matrix);
}

void GLShader::SetUniform1f(uint32_t location, float value)
{
    glUniform1f(location, value);
}

void GLShader::SetUniform1fv(uint32_t location, float* value, int32_t count)
{
    glUniform1fv(location, count, value);
}

void GLShader::SetUniform1i(uint32_t location, int32_t value)
{
    glUniform1i(location, value);
}

void GLShader::SetUniform1ui(uint32_t location, uint32_t value)
{
    glUniform1ui(location, value);
}

void GLShader::SetUniform1iv(uint32_t location, int32_t* value, int32_t count)
{
    glUniform1iv(location, count, value);
}

void GLShader::SetUniform2f(uint32_t location, const Eigen::Vector2f& vector)
{
    glUniform2f(location, vector.x(), vector.y());
}

void GLShader::SetUniform3f(uint32_t location, const Eigen::Vector3f & vector)
{
    glUniform3f(location, vector.x(), vector.y(), vector.z());
}

void GLShader::SetUniform4f(uint32_t location, const Eigen::Vector4f & vector)
{
    glUniform4f(location, vector.x(), vector.y(), vector.z(), vector.w());
}

void GLShader::SetUniformMat3(uint32_t location, const Eigen::Matrix3f& matrix)
{
    glUniformMatrix3fv(location, 1, GL_FALSE /*GLTRUE*/, matrix.data()); // &matrix.values[0]));
}

void GLShader::SetUniformMat4(uint32_t location, const Eigen::Matrix4f& matrix)
{
    glUniformMatrix4fv(location, 1, GL_FALSE /*GLTRUE*/, matrix.data());
}

void GLShader::SetUniformMat4Array(uint32_t location, uint32_t count, const Eigen::Matrix4f& matrix)
{
    glUniformMatrix4fv(location, count, GL_FALSE /*GLTRUE*/, matrix.data());
}

void GLShader::LoadFromData(const uint32_t* data, uint32_t size, ShaderType type, std::map<ShaderType, String>& sources)
{
    std::vector<unsigned int> spv(data, data + size / sizeof(unsigned int));

//    spirv_cross::CompilerGLSL* glsl = new spirv_cross::CompilerGLSL(std::move(spv));
//
//    // The SPIR-V is now parsed, and we can perform reflection on it.
//    spirv_cross::ShaderResources resources = glsl->get_shader_resources();
//
//    if(type == ShaderType::VERTEX)
//    {
//        uint32_t stride = 0;
//        for(const spirv_cross::Resource& resource : resources.stage_inputs)
//        {
//            const spirv_cross::SPIRType& InputType = glsl->get_type(resource.type_id);
//            // Switch to GL layout
//            PushTypeToBuffer(InputType, m_Layout);
//            stride += GetStrideFromOpenGLFormat(0); // InputType.width * InputType.vecsize / 8;
//        }
//    }
//
//    // Get all sampled images in the shader.
//    for(auto& resource : resources.sampled_images)
//    {
//        uint32_t set     = glsl->get_decoration(resource.id, spv::DecorationDescriptorSet);
//        uint32_t binding = glsl->get_decoration(resource.id, spv::DecorationBinding);
//
//        // Modify the decoration to prepare it for GLSL.
//        // glsl->unset_decoration(resource.id, spv::DecorationDescriptorSet);
//
//        // Some arbitrary remapping if we want.
//        // glsl->set_decoration(resource.id, spv::DecorationBinding, set * 16 + binding);
//
//        auto& descriptorInfo  = m_DescriptorInfos[set];
//        auto& descriptor      = descriptorInfo.descriptors.emplace_back();
//        descriptor.binding    = binding;
//        descriptor.name       = resource.name;
//        descriptor.shaderType = type;
//        descriptor.type       = Graphics::DescriptorType::IMAGE_SAMPLER;
//    }

    //                for(auto const& image : resources.separate_images)
    //                {
    //                    auto set { glsl->get_decoration(image.id, spv::Decoration::DecorationDescriptorSet) };
    //                    glsl->set_decoration(image.id, spv::Decoration::DecorationDescriptorSet, DESCRIPTOR_TABLE_INITIAL_SPACE + 2 * set);
    //                }
    //                for(auto const& input : resources.subpass_inputs)
    //                {
    //                    auto set { glsl->get_decoration(input.id, spv::Decoration::DecorationDescriptorSet) };
    //                    glsl->set_decoration(input.id, spv::Decoration::DecorationDescriptorSet, DESCRIPTOR_TABLE_INITIAL_SPACE + 2 * set);
    //                }
//    for(auto const& uniform_buffer : resources.uniform_buffers)
//    {
//        auto set { glsl->get_decoration(uniform_buffer.id, spv::Decoration::DecorationDescriptorSet) };
//        glsl->set_decoration(uniform_buffer.id, spv::Decoration::DecorationDescriptorSet, DESCRIPTOR_TABLE_INITIAL_SPACE + 2 * set);
//
//        uint32_t binding = glsl->get_decoration(uniform_buffer.id, spv::DecorationBinding);
//        auto& bufferType = glsl->get_type(uniform_buffer.type_id);
//
//        auto bufferSize = glsl->get_declared_struct_size(bufferType);
//        int memberCount = (int)bufferType.member_types.size();
//
//        auto& descriptorInfo  = m_DescriptorInfos[set];
//        auto& descriptor      = descriptorInfo.descriptors.emplace_back();
//        descriptor.binding    = binding;
//        descriptor.size       = (uint32_t)bufferSize;
//        descriptor.name       = uniform_buffer.name;
//        descriptor.offset     = 0;
//        descriptor.shaderType = type;
//        descriptor.type       = Graphics::DescriptorType::UNIFORM_BUFFER;
//        descriptor.buffer     = nullptr;
//
//        for(int i = 0; i < memberCount; i++)
//        {
//            auto type              = glsl->get_type(bufferType.member_types[i]);
//            const auto& memberName = glsl->get_member_name(bufferType.self, i);
//            auto size              = glsl->get_declared_struct_member_size(bufferType, i);
//            auto offset            = glsl->type_struct_member_offset(bufferType, i);
//
//            String uniformName = uniform_buffer.name + "." + memberName;
//
//            auto& member    = descriptor.m_Members.emplace_back();
//            member.size     = (uint32_t)size;
//            member.offset   = offset;
//            member.type     = SPIRVTypeToLumosDataType(type);
//            member.fullName = uniformName;
//            member.name     = memberName;
//        }
//    }
//    //                for(auto const& storage_buffer : resources.storage_buffers)
//    //                {
//    //                    auto set { glsl->get_decoration(storage_buffer.id, spv::Decoration::DecorationDescriptorSet) };
//    //                    glsl->set_decoration(storage_buffer.id, spv::Decoration::DecorationDescriptorSet, DESCRIPTOR_TABLE_INITIAL_SPACE + 2 * set);
//    //                }
//    //                for(auto const& storage_image : resources.storage_images)
//    //                {
//    //                    auto set { glsl->get_decoration(storage_image.id, spv::Decoration::DecorationDescriptorSet) };
//    //                    glsl->set_decoration(storage_image.id, spv::Decoration::DecorationDescriptorSet, DESCRIPTOR_TABLE_INITIAL_SPACE + 2 * set);
//    //                }
//    //
//    //                for(auto const& sampler : resources.separate_samplers)
//    //                {
//    //                    auto set { glsl->get_decoration(sampler.id, spv::Decoration::DecorationDescriptorSet) };
//    //                    glsl->set_decoration(sampler.id, spv::Decoration::DecorationDescriptorSet, DESCRIPTOR_TABLE_INITIAL_SPACE + 2 * set + 1);
//    //                }
//
//    for(auto& u : resources.push_constant_buffers)
//    {
//        // uint32_t set = glsl->get_decoration(u.id, spv::DecorationDescriptorSet);
//        // uint32_t binding = glsl->get_decoration(u.id, spv::DecorationBinding);
//
//        auto& pushConstantType = glsl->get_type(u.type_id);
//        auto name              = glsl->get_name(u.id);
//
//        auto ranges = glsl->get_active_buffer_ranges(u.id);
//
//        uint32_t rangeSizes = 0;
//        for(auto& range : ranges)
//        {
//            rangeSizes += uint32_t(range.range);
//        }
//
//        auto& bufferType = glsl->get_type(u.base_type_id);
//        auto bufferSize  = glsl->get_declared_struct_size(bufferType);
//        int memberCount  = (int)bufferType.member_types.size();
//
//        m_PushConstants.push_back({ rangeSizes, type });
//        m_PushConstants.back().data = new uint8_t[size];
//
//        for(int i = 0; i < memberCount; i++)
//        {
//            auto type              = glsl->get_type(bufferType.member_types[i]);
//            const auto& memberName = glsl->get_member_name(bufferType.self, i);
//            auto size              = glsl->get_declared_struct_member_size(bufferType, i);
//            auto offset            = glsl->type_struct_member_offset(bufferType, i);
//
//            String uniformName = u.name + "." + memberName;
//
//            auto& member    = m_PushConstants.back().m_Members.emplace_back();
//            member.size     = (uint32_t)size;
//            member.offset   = offset;
//            member.type     = SPIRVTypeToLumosDataType(type);
//            member.fullName = uniformName;
//            member.name     = memberName;
//        }
//    }
//
//    int imageCount[16]  = { 0 };
//    int bufferCount[16] = { 0 };
//
//    for(int i = 0; i < m_DescriptorInfos.size(); i++)
//    {
//        auto& descriptorInfo = m_DescriptorInfos[i];
//        for(auto& descriptor : descriptorInfo.descriptors)
//        {
//            if(descriptor.type == DescriptorType::IMAGE_SAMPLER)
//            {
//                imageCount[i]++;
//
//                if(i > 0)
//                    descriptor.binding = descriptor.binding + imageCount[i - 1];
//            }
//            else if(descriptor.type == DescriptorType::UNIFORM_BUFFER)
//            {
//                bufferCount[i]++;
//
//                if(i > 0)
//                    descriptor.binding = descriptor.binding + bufferCount[i - 1];
//            }
//        }
//    }
//
//    spirv_cross::CompilerGLSL::Options options;
//    options.version                              = 410;
//    options.es                                   = false;
//    options.vulkan_semantics                     = false;
//    options.separate_shader_objects              = false;
//    options.enable_420pack_extension             = false;
//    options.emit_push_constant_as_uniform_buffer = false;
//    glsl->set_common_options(options);
//
//    // Compile to GLSL, ready to give to GL driver.
//    String glslSource = glsl->compile();
//    sources[type]          = glslSource;
//
//    m_ShaderCompilers.push_back(glsl);
}

Shader* GLShader::CreateFuncGL(const String& filePath)
{
    String physicalPath;
//    Lumos::VFS::Get().ResolvePhysicalPath(filePath, physicalPath);
    GLShader* result = new GLShader(physicalPath);
    return result;
}

Shader* GLShader::CreateFromEmbeddedFuncGL(const uint32_t* vertData, uint32_t vertDataSize, const uint32_t* fragData, uint32_t fragDataSize)
{
    return new GLShader(vertData, vertDataSize, fragData, fragDataSize);
}

Shader* GLShader::CreateCompFromEmbeddedFuncGL(const uint32_t* compData, uint32_t compDataSize)
{
    return new GLShader(compData, compDataSize);
}

void GLShader::MakeDefault()
{
    CreateFunc                 = CreateFuncGL;
    CreateFuncFromEmbedded     = CreateFromEmbeddedFuncGL;
    CreateCompFuncFromEmbedded = CreateCompFromEmbeddedFuncGL;
}


}