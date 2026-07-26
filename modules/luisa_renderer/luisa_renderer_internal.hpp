#pragma once

#include "gobot/rendering/luisa_renderer_module_api.hpp"

#include <cuda.h>
#include <cudaGL.h>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(GOBOT_HAS_GSPLAT_INFERENCE)
namespace gsplat_inference {
struct Renderer;
}
#endif

namespace gobot::luisa_renderer {

using namespace luisa;
using namespace luisa::compute;

inline constexpr std::uint32_t kInvalidTexture =
        std::numeric_limits<std::uint32_t>::max();
inline constexpr std::uint32_t kMaxLights = 16;
inline constexpr std::uint32_t kRgbOutput = 1u << 0u;
inline constexpr std::uint32_t kDepthOutput = 1u << 1u;
inline constexpr std::uint32_t kNormalOutput = 1u << 2u;
inline constexpr std::uint32_t kInstanceOutput = 1u << 3u;
inline constexpr std::uint32_t kSemanticOutput = 1u << 4u;
inline constexpr std::uint32_t kAllProductOutputs = (1u << 5u) - 1u;
inline constexpr std::uint32_t kRgbVisibility = 1u << 0u;
inline constexpr std::uint32_t kShadowVisibility = 1u << 1u;
inline constexpr std::uint32_t kAovVisibility = 1u << 2u;

struct GpuVertex {
    float3 position;
    float3 normal;
    float4 tangent;
    float2 uv;
    float4 color;
};

struct GpuMaterial {
    float4 albedo;
    float4 emissive;
    float4 pbr;
    float4 options;
    uint4 textures0;
    uint4 textures1;
};

struct GpuLight {
    float4 position_type;
    float4 direction_range;
    float4 color_intensity;
    float4 spot_cosines;
};

struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

} // namespace gobot::luisa_renderer

LUISA_STRUCT(gobot::luisa_renderer::GpuVertex, position, normal, tangent, uv, color) {};
LUISA_STRUCT(gobot::luisa_renderer::GpuMaterial,
             albedo,
             emissive,
             pbr,
             options,
             textures0,
             textures1) {};
LUISA_STRUCT(gobot::luisa_renderer::GpuLight,
             position_type,
             direction_range,
             color_intensity,
             spot_cosines) {};
LUISA_STRUCT(gobot::luisa_renderer::Onb, tangent, binormal, normal) {
    [[nodiscard]] luisa::compute::Float3 to_world(
            luisa::compute::Expr<luisa::float3> value) const noexcept {
        return value.x * tangent + value.y * binormal + value.z * normal;
    }
};

namespace gobot::luisa_renderer {

class GobotBinaryIO;

SceneRendererCapabilities MakeCapabilities();
void SetError(char* destination, std::size_t size, const std::string& message);
std::string CudaError(CUresult result, const char* operation);
Device CreateCudaDevice(Context& context, const BinaryIO* binary_io);
float4x4 ToLuisaMatrix(const Matrix4& matrix);
std::array<float, 16> ToFloatMatrix(const Matrix4& matrix);
float3 ToFloat3(const Vector3& value);
float4 ToFloat4(const Color& value);
std::vector<float4> ConvertImage(const ImageStorageData& image);
Sampler ToSampler(const RenderTextureSnapshot& texture);

class CudaContextScope final {
public:
    explicit CudaContextScope(CUcontext context) noexcept;
    ~CudaContextScope() noexcept;

    CudaContextScope(const CudaContextScope&) = delete;
    CudaContextScope& operator=(const CudaContextScope&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept { return active_; }
    [[nodiscard]] CUresult GetResult() const noexcept { return result_; }

private:
    CUresult result_ = CUDA_ERROR_INVALID_CONTEXT;
    bool active_ = false;
};

class CudaGraphicsImage final {
public:
    CudaGraphicsImage() noexcept = default;
    explicit CudaGraphicsImage(CUcontext context) noexcept : context_(context) {}
    ~CudaGraphicsImage() noexcept;

    CudaGraphicsImage(const CudaGraphicsImage&) = delete;
    CudaGraphicsImage& operator=(const CudaGraphicsImage&) = delete;

    void SetContext(CUcontext context) noexcept { context_ = context; }
    bool Ensure(std::uint32_t texture, std::string* error);
    bool Reset(std::string* error = nullptr);

    [[nodiscard]] CUgraphicsResource Get() const noexcept { return resource_; }

private:
    CUcontext context_ = nullptr;
    CUgraphicsResource resource_ = nullptr;
    std::uint32_t texture_ = 0;
};

class CudaGraphicsMap final {
public:
    CudaGraphicsMap(CUgraphicsResource resource, CUstream stream) noexcept;
    ~CudaGraphicsMap() noexcept;

    CudaGraphicsMap(const CudaGraphicsMap&) = delete;
    CudaGraphicsMap& operator=(const CudaGraphicsMap&) = delete;

    bool GetArray(CUarray* destination, std::string* error);
    bool Unmap(std::string* error);

private:
    CUgraphicsResource resource_ = nullptr;
    CUstream stream_ = nullptr;
    CUresult map_result_ = CUDA_ERROR_INVALID_HANDLE;
    bool mapped_ = false;
};

class CudaEvent final {
public:
    CudaEvent() noexcept = default;
    explicit CudaEvent(CUcontext context) noexcept : context_(context) {}
    ~CudaEvent() noexcept;

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    void SetContext(CUcontext context) noexcept { context_ = context; }
    bool Record(CUstream stream, std::string* error);
    bool ElapsedMilliseconds(const CudaEvent& end,
                             double* milliseconds,
                             std::string* error) const;

private:
    bool Ensure(std::string* error);

    CUcontext context_ = nullptr;
    CUevent event_ = nullptr;
};

struct GeometryKey {
    std::uint64_t mesh_id = 0;
    std::uint64_t revision = 0;
    std::size_t surface = 0;
    bool operator==(const GeometryKey&) const = default;
};

struct GeometryKeyHash {
    std::size_t operator()(const GeometryKey& key) const;
};

struct TextureKey {
    std::uint64_t texture_id = 0;
    std::uint64_t texture_revision = 0;
    std::uint64_t image_id = 0;
    std::uint64_t image_revision = 0;
    bool operator==(const TextureKey&) const = default;
};

struct TextureKeyHash {
    std::size_t operator()(const TextureKey& key) const;
};

struct GeometryResource {
    Buffer<GpuVertex> vertices;
    Buffer<Triangle> triangles;
    luisa::compute::Mesh mesh;
};

struct TextureResource {
    luisa::compute::Image<float> image;
};

struct CudaRenderProductFrame {
    int width = 0;
    int height = 0;
    std::uint32_t output_mask = 0;
    Buffer<uint> rgb;
    Buffer<float> linear_depth;
    Buffer<float4> world_normal;
    Buffer<uint> instance_id;
    Buffer<uint> semantic_id;
    Buffer<uint> geometry_coverage;
};

#if defined(GOBOT_HAS_GSPLAT_INFERENCE)
struct GaussianRendererDeleter {
    CUcontext context = nullptr;
    void operator()(gsplat_inference::Renderer* renderer) const noexcept;
};
using GaussianRendererPtr =
        std::unique_ptr<gsplat_inference::Renderer, GaussianRendererDeleter>;
#endif

class LuisaRenderer final {
public:
    explicit LuisaRenderer(const std::string& module_directory);
    ~LuisaRenderer();

    LuisaRenderer(const LuisaRenderer&) = delete;
    LuisaRenderer& operator=(const LuisaRenderer&) = delete;

    [[nodiscard]] SceneRendererCapabilities Capabilities() const;
    void ResetAccumulation();

    LuisaRendererResult Render(const LuisaRendererTarget& target,
                               const RenderSceneSnapshot& snapshot,
                               const RenderViewSnapshot& view,
                               const SceneRendererSettings& settings,
                               SceneRendererStats* stats,
                               std::string* error);
    LuisaRendererResult CaptureRenderProduct(const RenderSceneSnapshot& snapshot,
                                             const RenderViewSnapshot& view,
                                             const LuisaRenderProductRequest& request,
                                             LuisaRenderProductFrame* result,
                                             std::string* error);
    LuisaRendererResult RenderGaussianBackground(const LuisaRendererTarget& target,
                                                 const RenderSceneSnapshot& snapshot,
                                                 const RenderViewSnapshot& view,
                                                 SceneRendererStats* stats,
                                                 std::string* error);
    bool ReadbackRenderProduct(CudaRenderProductFrame* frame,
                               std::uint32_t output,
                               void* destination,
                               std::size_t destination_size,
                               std::string* error);

private:
    using TraceShader = Shader2D<luisa::compute::Image<float>,
                                 luisa::compute::Image<uint>,
                                 luisa::compute::Image<float>,
                                 Accel,
                                 BindlessArray,
                                 BindlessArray,
                                 Buffer<GpuMaterial>,
                                 Buffer<GpuLight>,
                                 uint,
                                 uint,
                                 float3,
                                 float3,
                                 float,
                                 float4x4,
                                 float3,
                                 uint,
                                 uint>;
    using ClearShader = Shader2D<luisa::compute::Image<float>,
                                 luisa::compute::Image<float>>;
    using SeedShader = Shader2D<luisa::compute::Image<uint>, uint>;
    using ToneShader = Shader2D<luisa::compute::Image<float>,
                                luisa::compute::Image<float>,
                                Buffer<uint>,
                                float,
                                uint,
                                uint2>;
    using ProductShader = Shader2D<Buffer<uint>,
                                   Buffer<float>,
                                   Buffer<float4>,
                                   Buffer<uint>,
                                   Buffer<uint>,
                                   Buffer<uint>,
                                   Accel,
                                   BindlessArray,
                                   BindlessArray,
                                   Buffer<GpuMaterial>,
                                   Buffer<GpuLight>,
                                   Buffer<uint>,
                                   Buffer<uint>,
                                   Buffer<uint>,
                                   uint,
                                   uint,
                                   uint,
                                   uint,
                                   float3,
                                   float3,
                                   float,
                                   float,
                                   float,
                                   float4x4,
                                   float4x4,
                                   float3>;

    static std::string ResolveRuntimeDirectory(const std::string& module_directory);
    [[nodiscard]] CUcontext NativeContext() const noexcept;
    [[nodiscard]] CUstream NativeStream() const noexcept;
    bool SynchronizeStream(const char* operation, std::string* error);

    void CompileShaders();
    GeometryResource* EnsureGeometry(const VisualMeshRenderItem& item, std::string* error);
    std::uint32_t BindTexture(const RenderTextureSnapshot& texture);
    GpuMaterial MakeMaterial(const RenderMaterialSnapshot& material);
    bool RebuildTopology(const RenderSceneSnapshot& snapshot, std::string* error);
    void UpdateTransforms(const RenderSceneSnapshot& snapshot);
    void UpdateMaterialsAndLighting(const RenderSceneSnapshot& snapshot);
    bool SyncScene(const RenderSceneSnapshot& snapshot,
                   std::string* error,
                   bool allow_empty = false);
    bool EnsureFrameResources(int width, int height, std::string* error);
    bool Present(const LuisaRendererTarget& target, std::string* error);
    bool EnqueueGaussianToBuffer(const RenderSceneSnapshot& snapshot,
                                 const RenderViewSnapshot& view,
                                 int width,
                                 int height,
                                 void* output,
                                 const void* geometry_coverage,
                                 bool top_left_origin,
                                 std::string* error);

    std::string runtime_directory_;
    std::unique_ptr<Context> context_;
    std::unique_ptr<GobotBinaryIO> binary_io_;
    std::unique_ptr<Device> device_;
    std::unique_ptr<Stream> stream_;
    TraceShader trace_shader_;
    ClearShader clear_shader_;
    SeedShader seed_shader_;
    ToneShader tone_shader_;
    ProductShader product_shader_;

    std::unordered_map<GeometryKey, std::unique_ptr<GeometryResource>, GeometryKeyHash>
            geometry_cache_;
    std::unordered_map<TextureKey, std::unique_ptr<TextureResource>, TextureKeyHash>
            texture_cache_;
    std::vector<GeometryResource*> active_geometry_;
    Accel accel_;
    BindlessArray geometry_heap_;
    BindlessArray texture_heap_;
    Buffer<GpuMaterial> materials_;
    Buffer<GpuLight> lights_;
    Buffer<uint> instance_ids_;
    Buffer<uint> semantic_ids_;
    Buffer<uint> rgb_modes_;
    Buffer<uint> dummy_uint_;
    Buffer<float> dummy_float_;
    Buffer<float4> dummy_float4_;
    std::size_t active_light_count_ = 0;
    std::uint32_t next_texture_slot_ = 0;
    std::uint32_t environment_texture_slot_ = kInvalidTexture;

    luisa::compute::Image<float> accumulation_;
    luisa::compute::Image<float> guide_;
    luisa::compute::Image<uint> seeds_;
    Buffer<uint> presentation_;
    int frame_width_ = 0;
    int frame_height_ = 0;
    bool accumulation_valid_ = false;
    std::uint64_t accumulated_samples_ = 0;
    std::uint64_t stable_frame_count_ = 0;
    double previous_render_ms_ = 0.0;

    std::uint64_t last_topology_ = 0;
    std::uint64_t last_geometry_ = 0;
    std::uint64_t last_transforms_ = 0;
    std::uint64_t last_materials_ = 0;
    std::uint64_t last_lighting_ = 0;
    std::uint64_t last_combined_ = 0;

    CudaGraphicsImage interop_;
    CudaEvent render_start_event_;
    CudaEvent render_end_event_;
    CudaEvent presentation_end_event_;
    CudaEvent gaussian_start_event_;
    CudaEvent gaussian_end_event_;

#if defined(GOBOT_HAS_GSPLAT_INFERENCE)
    GaussianRendererPtr gaussian_renderer_{nullptr, GaussianRendererDeleter{}};
    std::uint64_t gaussian_resource_id_ = 0;
    std::uint64_t gaussian_resource_revision_ = 0;
#endif
};

} // namespace gobot::luisa_renderer
