/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include "gobot/render/texture.hpp"

namespace gobot {

class GOBOT_EXPORT GLTexture2D : public Texture2D
{
public:
    GLTexture2D(TextureDesc parameters, uint32_t width, uint32_t height);
    GLTexture2D(uint32_t width, uint32_t height, void* data, TextureDesc parameters = TextureDesc(), TextureLoadOptions loadOptions = TextureLoadOptions());
    GLTexture2D(const std::string& name, const std::string& filename, TextureDesc parameters = TextureDesc(), TextureLoadOptions loadOptions = TextureLoadOptions());
    ~GLTexture2D();

    void Bind(uint32_t slot = 0) const override;
    void Unbind(uint32_t slot = 0) const override;

    virtual void SetData(const void* pixels) override;

    virtual void* GetHandle() const override
    {
        return (void*)(size_t)m_Handle;
    }

    inline uint32_t GetWidth(uint32_t mip) const override
    {
        return m_Width >> mip;
    }

    inline uint32_t GetHeight(uint32_t mip) const override
    {
        return m_Height >> mip;
    }

    inline const std::string& GetName() const override
    {
        return m_Name;
    }
    inline const std::string& GetFilepath() const override
    {
        return m_FileName;
    }

    void Resize(uint32_t width, uint32_t height) override;
    void BuildTexture();

    uint8_t* LoadTextureData();
    uint32_t LoadTexture(void* data);

    TextureType GetType() const override
    {
        return TextureType::COLOUR;
    }

    RHIFormat GetFormat() const override
    {
        return m_Format;
    }

    static void MakeDefault();

protected:
    static Texture2D* CreateFuncGL(TextureDesc parameters, uint32_t width, uint32_t height);
    static Texture2D* CreateFromSourceFuncGL(uint32_t, uint32_t, void*, TextureDesc, TextureLoadOptions);
    static Texture2D* CreateFromFileFuncGL(const std::string&, const std::string&, TextureDesc, TextureLoadOptions);

private:
    uint32_t Load(void* data);

    std::string m_Name;
    std::string m_FileName;
    uint32_t m_Handle;
    uint32_t m_Width, m_Height;
    TextureDesc m_Parameters;
    TextureLoadOptions m_LoadOptions;
    RHIFormat m_Format;
    bool isHDR = false;
};

class GLTextureCube : public TextureCube
{
public:
    GLTextureCube(uint32_t size);
    GLTextureCube(const std::string& filepath);
    GLTextureCube(const std::string* files);
    GLTextureCube(const std::string* files, uint32_t mips, TextureDesc params, TextureLoadOptions loadOptions);
    ~GLTextureCube();

    inline void* GetHandle() const override
    {
        return (void*)(size_t)m_Handle;
    }

    void Bind(uint32_t slot = 0) const override;
    void Unbind(uint32_t slot = 0) const override;

    uint32_t GetMipMapLevels() const override
    {
        return m_NumMips;
    }

    inline uint32_t GetSize() const override
    {
        return m_Size;
    }
    inline const std::string& GetName() const override
    {
        return m_Name;
    }
    inline const std::string& GetFilepath() const override
    {
        return m_Files[0];
    }

    inline uint32_t GetWidth(uint32_t mip) const override
    {
        return m_Width >> mip;
    }

    inline uint32_t GetHeight(uint32_t mip) const override
    {
        return m_Height >> mip;
    }

    TextureType GetType() const override
    {
        return TextureType::CUBE;
    }

    RHIFormat GetFormat() const override
    {
        return m_Format;
    }

    static void MakeDefault();

protected:
    static TextureCube* CreateFuncGL(uint32_t, void*, bool);
    static TextureCube* CreateFromFileFuncGL(const std::string& filepath);
    static TextureCube* CreateFromFilesFuncGL(const std::string* files);
    static TextureCube* CreateFromVCrossFuncGL(const std::string* files, uint32_t mips, TextureDesc params, TextureLoadOptions loadOptions);

private:
    static uint32_t LoadFromSingleFile();
    uint32_t LoadFromMultipleFiles();
    uint32_t LoadFromVCross(uint32_t mips);

    uint32_t m_Handle;
    uint32_t m_Width, m_Height, m_Size;
    std::string m_Name;
    std::string m_Files[MAX_MIPS];
    uint32_t m_Bits;
    uint32_t m_NumMips;
    RHIFormat m_Format;
    TextureDesc m_Parameters;
    TextureLoadOptions m_LoadOptions;
};

class GLTextureDepth : public TextureDepth
{
public:
    GLTextureDepth(uint32_t width, uint32_t height);
    ~GLTextureDepth();

    void Bind(uint32_t slot = 0) const override;
    void Unbind(uint32_t slot = 0) const override;
    void Resize(uint32_t width, uint32_t height) override;

    inline void* GetHandle() const override
    {
        return (void*)(size_t)m_Handle;
    }

    inline const std::string& GetName() const override
    {
        return m_Name;
    }
    inline const std::string& GetFilepath() const override
    {
        return m_Name;
    }

    inline uint32_t GetWidth(uint32_t mip) const override
    {
        return m_Width >> mip;
    }

    inline uint32_t GetHeight(uint32_t mip) const override
    {
        return m_Height >> mip;
    }

    TextureType GetType() const override
    {
        return TextureType::DEPTH;
    }

    RHIFormat GetFormat() const override
    {
        return m_Format;
    }

    static void MakeDefault();

protected:
    static TextureDepth* CreateFuncGL(uint32_t, uint32_t);

    void Init();

    std::string m_Name;
    uint32_t m_Handle;
    uint32_t m_Width, m_Height;
    RHIFormat m_Format;
};

class GLTextureDepthArray : public TextureDepthArray
{
private:
    std::string m_Name;
    uint32_t m_Handle;
    uint32_t m_Width, m_Height, m_Count;

public:
    GLTextureDepthArray(uint32_t width, uint32_t height, uint32_t count);
    ~GLTextureDepthArray();

    void Bind(uint32_t slot = 0) const override;
    void Unbind(uint32_t slot = 0) const override;
    void Resize(uint32_t width, uint32_t height, uint32_t count) override;

    inline void* GetHandle() const override
    {
        return (void*)(size_t)m_Handle;
    }

    inline const std::string& GetName() const override
    {
        return m_Name;
    }

    inline const std::string& GetFilepath() const override
    {
        return m_Name;
    }

    inline uint32_t GetWidth(uint32_t mip) const override
    {
        return m_Width >> mip;
    }

    inline uint32_t GetHeight(uint32_t mip) const override
    {
        return m_Height >> mip;
    }

    inline void SetCount(uint32_t count)
    {
        m_Count = count;
    }

    TextureType GetType() const override
    {
        return TextureType::DEPTHARRAY;
    }

    RHIFormat GetFormat() const override
    {
        return m_Format;
    }

    uint32_t GetCount() const { return m_Count; }

    void Init() override;

    static void MakeDefault();

protected:
    static TextureDepthArray* CreateFuncGL(uint32_t, uint32_t, uint32_t);
    RHIFormat m_Format;
};


}