#include "test_render_backend.hpp"

#include "gobot/core/rid_owner.hpp"
#include "gobot/rendering/material_storage.hpp"
#include "gobot/rendering/mesh_storage.hpp"
#include "gobot/rendering/render_utilities.hpp"
#include "gobot/rendering/renderer_compositor.hpp"
#include "gobot/rendering/renderer_debug_draw.hpp"
#include "gobot/rendering/renderer_scene_render.hpp"
#include "gobot/rendering/texture_storage.hpp"

#include <string>
#include <unordered_map>

namespace gobot::test {
namespace {

struct TestMesh {};
struct TestMaterial {};
struct TestShader {
    std::string code;
};
struct TestShaderProgram {};
struct TestTexture {};
struct TestRenderTarget {
    int width = 0;
    int height = 0;
    uint32_t view_count = 0;
};

Rasterizer* CreateTestRasterizer();

class TestMeshStorage final : public MeshStorage {
public:
    RID MeshAllocate() override { return meshes_.AllocateRID(); }
    void MeshInitialize(const RID& p_rid) override { meshes_.InitializeRID(p_rid); }
    void MeshSetBox(const RID&, const Vector3&) override {}
    void MeshSetSurface(const RID&,
                        const std::vector<Vector3>&,
                        const std::vector<uint32_t>&,
                        const std::vector<Vector3>&) override {}
    void MeshSetCylinder(const RID&, RealType, RealType, int) override {}
    void MeshSetSphere(const RID&, RealType, int, int) override {}
    bool OwnsMesh(const RID& p_rid) const override { return meshes_.Owns(p_rid); }
    void MeshFree(const RID& p_rid) override { meshes_.Free(p_rid); }

private:
    mutable RID_Owner<TestMesh, true> meshes_;
};

class TestMaterialStorage final : public MaterialStorage {
public:
    RID ShaderAllocate() override { return shaders_.AllocateRID(); }
    void ShaderInitialize(RID p_rid, ShaderType) override { shaders_.InitializeRID(p_rid); }
    void ShaderSetCode(RID p_shader,
                       const std::string& p_code,
                       const std::string&,
                       const std::string&) override {
        if (auto* shader = shaders_.GetOrNull(p_shader)) {
            shader->code = p_code;
        }
    }
    std::string ShaderGetCode(RID p_shader) const override {
        const auto* shader = shaders_.GetOrNull(p_shader);
        return shader == nullptr ? std::string() : shader->code;
    }
    void ShaderFree(RID p_rid) override { shaders_.Free(p_rid); }

    RID ShaderProgramAllocate() override { return shader_programs_.AllocateRID(); }
    void ShaderProgramInitialize(RID p_rid) override { shader_programs_.InitializeRID(p_rid); }
    void ShaderProgramFree(RID p_rid) override { shader_programs_.Free(p_rid); }
    void ShaderProgramSetRasterizerShader(RID, RID, RID, RID, RID, RID, const std::string&) override {}
    void ShaderProgramSetComputeShader(RID, RID, const std::string&) override {}

    RID MaterialAllocate() override { return materials_.AllocateRID(); }
    void MaterialInitialize(RID p_rid) override { materials_.InitializeRID(p_rid); }
    void MaterialFree(RID p_rid) override { materials_.Free(p_rid); }

private:
    mutable RID_Owner<TestShader, true> shaders_;
    RID_Owner<TestShaderProgram, true> shader_programs_;
    RID_Owner<TestMaterial, true> materials_;
};

class TestTextureStorage final : public RendererTextureStorage {
public:
    RID RenderTargetCreate() override {
        RID rid = render_targets_.AllocateRID();
        render_targets_.InitializeRID(rid);
        return rid;
    }
    void RenderTargetFree(RID p_rid) override { render_targets_.Free(p_rid); }
    void RenderTargetSetSize(RID p_render_target, int p_width, int p_height, uint32_t p_view_count) override {
        if (auto* target = render_targets_.GetOrNull(p_render_target)) {
            target->width = p_width;
            target->height = p_height;
            target->view_count = p_view_count;
        }
    }
    void* GetRenderTargetColorTextureNativeHandle(RID) override { return nullptr; }

    RID TextureAllocate() override {
        RID rid = textures_.AllocateRID();
        textures_.InitializeRID(rid);
        return rid;
    }
    void TextureFree(RID p_rid) override { textures_.Free(p_rid); }

private:
    RID_Owner<TestRenderTarget, true> render_targets_;
    RID_Owner<TestTexture, true> textures_;
};

class TestSceneRender final : public RendererSceneRender {
public:
    void RenderScene(const RID&, const Node*, const Camera3D*) override {}
};

class TestDebugDraw final : public RendererDebugDraw {
public:
    void RenderEditorDebug(const RID&, const Camera3D*, const Node*) override {}
};

class TestUtilities final : public RendererUtilities {
public:
    bool Free(RID) override { return false; }
};

class TestRasterizer final : public Rasterizer {
public:
    static void MakeCurrent() {
        CreateFunc = CreateTestRasterizer;
    }

    RendererSceneRender* GetScene() override { return &scene_; }
    RendererTextureStorage* GetTextureStorage() override { return &textures_; }
    MaterialStorage* GetMaterialStorage() override { return &materials_; }
    MeshStorage* GetMeshStorage() override { return &meshes_; }
    RendererDebugDraw* GetDebugDraw() override { return &debug_draw_; }
    RendererUtilities* GetUtilities() override { return &utilities_; }

    void Initialize() override {}
    void BeginFrame(double) override {}
    void EndFrame(bool) override {}
    void Finalize() override {}

private:
    TestSceneRender scene_;
    TestTextureStorage textures_;
    TestMaterialStorage materials_;
    TestMeshStorage meshes_;
    TestDebugDraw debug_draw_;
    TestUtilities utilities_;
};

Rasterizer* CreateTestRasterizer() {
    return new TestRasterizer();
}

} // namespace

void RegisterTestRasterizer() {
    TestRasterizer::MakeCurrent();
}

} // namespace gobot::test
