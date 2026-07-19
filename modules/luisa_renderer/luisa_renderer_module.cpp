/*
 * Optional LuisaCompute CUDA path tracer for Gobot's OpenGL editor viewport.
 * This module is intentionally outside src/gobot so default builds never link it.
 */

#include "gobot/rendering/luisa_renderer_module_api.hpp"

#include <glad/glad.h>
#include <cuda.h>
#include <cudaGL.h>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace luisa;
using namespace luisa::compute;

constexpr std::uint32_t kInvalidTexture = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t kMaxLights = 16;

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

LUISA_STRUCT(GpuVertex, position, normal, tangent, uv, color) {};
LUISA_STRUCT(GpuMaterial, albedo, emissive, pbr, options, textures0, textures1) {};
LUISA_STRUCT(GpuLight, position_type, direction_range, color_intensity, spot_cosines) {};
LUISA_STRUCT(Onb, tangent, binormal, normal) {
    [[nodiscard]] Float3 to_world(Expr<float3> value) const noexcept {
        return value.x * tangent + value.y * binormal + value.z * normal;
    }
};

namespace {

gobot::SceneRendererCapabilities MakeCapabilities() {
    gobot::SceneRendererCapabilities capabilities;
    capabilities.ray_tracing_available = true;
    capabilities.realtime = true;
    capabilities.progressive = true;
    capabilities.denoise = true;
    capabilities.direct_presentation_interop = true;
    capabilities.backend_name = "LuisaCompute CUDA";
    capabilities.status =
            "LuisaCompute CUDA/OptiX (CUDA " GOBOT_LUISA_CUDA_TOOLKIT_VERSION
            ") with direct CUDA-OpenGL presentation";
    return capabilities;
}

void SetError(char* destination, std::size_t size, const std::string& message) {
    if (destination == nullptr || size == 0) {
        return;
    }
    const std::size_t count = std::min(size - 1, message.size());
    std::memcpy(destination, message.data(), count);
    destination[count] = '\0';
}

std::string CudaError(CUresult result, const char* operation) {
    const char* name = nullptr;
    const char* description = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);
    return std::string(operation) + " failed: " + (name != nullptr ? name : "CUDA_ERROR") +
           " (" + (description != nullptr ? description : "unknown") + ")";
}

class FileBinaryStream final : public BinaryStream {
public:
    explicit FileBinaryStream(std::vector<std::byte> data) : data_(std::move(data)) {}

    [[nodiscard]] std::size_t length() const noexcept override { return data_.size(); }
    [[nodiscard]] std::size_t pos() const noexcept override { return position_; }

    void read(luisa::span<std::byte> destination) noexcept override {
        const std::size_t count = std::min(destination.size(), data_.size() - position_);
        if (count != 0u) {
            std::memcpy(destination.data(), data_.data() + position_, count);
            position_ += count;
        }
    }

private:
    std::vector<std::byte> data_;
    std::size_t position_ = 0;
};

class GobotBinaryIO final : public BinaryIO {
public:
    explicit GobotBinaryIO(std::filesystem::path runtime_directory)
        : runtime_directory_(std::move(runtime_directory)),
          cache_directory_(ResolveCacheDirectory()),
          bytecode_directory_(cache_directory_ / "bytecode"),
          internal_directory_(cache_directory_ / "internal") {
        std::error_code error;
        std::filesystem::create_directories(bytecode_directory_, error);
        error.clear();
        std::filesystem::create_directories(internal_directory_, error);
    }

    void clear_shader_cache() const noexcept override {
        std::error_code error;
        std::filesystem::remove_all(cache_directory_, error);
        error.clear();
        std::filesystem::create_directories(bytecode_directory_, error);
        error.clear();
        std::filesystem::create_directories(internal_directory_, error);
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream> read_shader_bytecode(
            luisa::string_view name) const noexcept override {
        const std::filesystem::path path{name};
        if (path.is_absolute()) {
            return Read(path);
        }
        if (auto stream = Read(runtime_directory_ / path); stream != nullptr) {
            return stream;
        }
        return Read(bytecode_directory_ / path);
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream> read_shader_cache(
            luisa::string_view name) const noexcept override {
        return Read(Resolve(cache_directory_, name));
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream> read_shader_source(
            luisa::string_view name) const noexcept override {
        return read_shader_cache(name);
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream> read_internal_shader(
            luisa::string_view name) const noexcept override {
        return Read(Resolve(internal_directory_, name));
    }

    luisa::filesystem::path write_shader_bytecode(
            luisa::string_view name,
            luisa::span<const std::byte> data) const noexcept override {
        return Write(Resolve(bytecode_directory_, name), data);
    }

    luisa::filesystem::path write_shader_cache(
            luisa::string_view name,
            luisa::span<const std::byte> data) const noexcept override {
        return Write(Resolve(cache_directory_, name), data);
    }

    luisa::filesystem::path write_shader_source(
            luisa::string_view name,
            luisa::span<const std::byte> data) const noexcept override {
        return write_shader_cache(name, data);
    }

    luisa::filesystem::path write_internal_shader(
            luisa::string_view name,
            luisa::span<const std::byte> data) const noexcept override {
        return Write(Resolve(internal_directory_, name), data);
    }

private:
    static std::filesystem::path ResolveCacheDirectory() {
        std::filesystem::path base;
        if (const char* xdg_cache = std::getenv("XDG_CACHE_HOME");
            xdg_cache != nullptr && xdg_cache[0] != '\0') {
            base = xdg_cache;
        } else if (const char* home = std::getenv("HOME");
                   home != nullptr && home[0] != '\0') {
            base = std::filesystem::path(home) / ".cache";
        } else {
            std::error_code error;
            base = std::filesystem::temp_directory_path(error);
            if (error) {
                base = "/tmp";
            }
        }
        return base / "gobot" / "luisa" / "v1";
    }

    static std::filesystem::path Resolve(const std::filesystem::path& directory,
                                         luisa::string_view name) {
        const std::filesystem::path path{name};
        return path.is_absolute() ? path : directory / path;
    }

    static luisa::unique_ptr<BinaryStream> Read(const std::filesystem::path& path) noexcept {
        try {
            std::ifstream input(path, std::ios::binary | std::ios::ate);
            if (!input) {
                return nullptr;
            }
            const std::streamsize size = input.tellg();
            if (size < 0) {
                return nullptr;
            }
            input.seekg(0, std::ios::beg);
            std::vector<std::byte> data(static_cast<std::size_t>(size));
            if (size != 0 && !input.read(reinterpret_cast<char*>(data.data()), size)) {
                return nullptr;
            }
            return luisa::make_unique<FileBinaryStream>(std::move(data));
        } catch (...) {
            return nullptr;
        }
    }

    static luisa::filesystem::path Write(const std::filesystem::path& path,
                                         luisa::span<const std::byte> data) noexcept {
        try {
            std::error_code error;
            std::filesystem::create_directories(path.parent_path(), error);
            if (error) {
                return {};
            }
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            if (!output) {
                return {};
            }
            output.write(reinterpret_cast<const char*>(data.data()),
                         static_cast<std::streamsize>(data.size()));
            return output ? path : luisa::filesystem::path{};
        } catch (...) {
            return {};
        }
    }

    std::filesystem::path runtime_directory_;
    std::filesystem::path cache_directory_;
    std::filesystem::path bytecode_directory_;
    std::filesystem::path internal_directory_;
};

std::size_t CurrentOpenGLCudaDeviceIndex() {
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(CudaError(result, "cuInit"));
    }

    std::array<CUdevice, 8> devices{};
    unsigned int device_count = 0;
    result = cuGLGetDevices(&device_count,
                            devices.data(),
                            static_cast<unsigned int>(devices.size()),
                            CU_GL_DEVICE_LIST_ALL);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(CudaError(result, "cuGLGetDevices"));
    }
    if (device_count == 0) {
        throw std::runtime_error(
                "The current OpenGL context is not associated with a CUDA device.");
    }
    return static_cast<std::size_t>(devices.front());
}

Device CreateCudaDevice(Context& context, const BinaryIO* binary_io) {
    DeviceConfig config;
    config.binary_io = binary_io;
    config.device_index = CurrentOpenGLCudaDeviceIndex();
    return context.create_device("cuda", &config);
}

float4x4 ToLuisaMatrix(const gobot::Matrix4& matrix) {
    return make_float4x4(
            make_float4(matrix(0, 0), matrix(1, 0), matrix(2, 0), matrix(3, 0)),
            make_float4(matrix(0, 1), matrix(1, 1), matrix(2, 1), matrix(3, 1)),
            make_float4(matrix(0, 2), matrix(1, 2), matrix(2, 2), matrix(3, 2)),
            make_float4(matrix(0, 3), matrix(1, 3), matrix(2, 3), matrix(3, 3)));
}

float3 ToFloat3(const gobot::Vector3& value) {
    return make_float3(static_cast<float>(value.x()),
                       static_cast<float>(value.y()),
                       static_cast<float>(value.z()));
}

float4 ToFloat4(const gobot::Color& value) {
    return make_float4(value.red(), value.green(), value.blue(), value.alpha());
}

float ReadFloat(const std::uint8_t* data) {
    float value = 0.0f;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

std::vector<float4> ConvertImage(const gobot::ImageStorageData& image) {
    const std::size_t pixel_count = static_cast<std::size_t>(image.width) * image.height;
    std::vector<float4> pixels(pixel_count, make_float4(1.0f));
    auto byte_channel = [&](std::size_t pixel, std::size_t channel, std::size_t channels) {
        return static_cast<float>(image.data[pixel * channels + channel]) / 255.0f;
    };
    auto float_channel = [&](std::size_t pixel, std::size_t channel, std::size_t channels) {
        return ReadFloat(image.data.data() + (pixel * channels + channel) * sizeof(float));
    };

    for (std::size_t i = 0; i < pixel_count; ++i) {
        switch (image.format) {
            case gobot::ImageFormat::L8: {
                const float value = byte_channel(i, 0, 1);
                pixels[i] = make_float4(value, value, value, 1.0f);
                break;
            }
            case gobot::ImageFormat::LA8: {
                const float value = byte_channel(i, 0, 2);
                pixels[i] = make_float4(value, value, value, byte_channel(i, 1, 2));
                break;
            }
            case gobot::ImageFormat::R8:
                pixels[i] = make_float4(byte_channel(i, 0, 1), 0.0f, 0.0f, 1.0f);
                break;
            case gobot::ImageFormat::RG8:
                pixels[i] = make_float4(byte_channel(i, 0, 2), byte_channel(i, 1, 2), 0.0f, 1.0f);
                break;
            case gobot::ImageFormat::RGB8:
                pixels[i] = make_float4(byte_channel(i, 0, 3),
                                        byte_channel(i, 1, 3),
                                        byte_channel(i, 2, 3),
                                        1.0f);
                break;
            case gobot::ImageFormat::RGBA8:
                pixels[i] = make_float4(byte_channel(i, 0, 4),
                                        byte_channel(i, 1, 4),
                                        byte_channel(i, 2, 4),
                                        byte_channel(i, 3, 4));
                break;
            case gobot::ImageFormat::RF: {
                const float value = float_channel(i, 0, 1);
                pixels[i] = make_float4(value, value, value, 1.0f);
                break;
            }
            case gobot::ImageFormat::RGF:
                pixels[i] = make_float4(float_channel(i, 0, 2), float_channel(i, 1, 2), 0.0f, 1.0f);
                break;
            case gobot::ImageFormat::RGBF:
                pixels[i] = make_float4(float_channel(i, 0, 3),
                                        float_channel(i, 1, 3),
                                        float_channel(i, 2, 3),
                                        1.0f);
                break;
            case gobot::ImageFormat::RGBAF:
                pixels[i] = make_float4(float_channel(i, 0, 4),
                                        float_channel(i, 1, 4),
                                        float_channel(i, 2, 4),
                                        float_channel(i, 3, 4));
                break;
            default:
                return {};
        }
    }
    return pixels;
}

Sampler ToSampler(const gobot::RenderTextureSnapshot& texture) {
    const auto filter = texture.min_filter == gobot::TextureFilter::Nearest
                                ? Sampler::Filter::POINT
                                : Sampler::Filter::LINEAR_LINEAR;
    Sampler::Address address = Sampler::Address::REPEAT;
    if (texture.wrap_u == gobot::TextureWrap::ClampToEdge ||
        texture.wrap_v == gobot::TextureWrap::ClampToEdge) {
        address = Sampler::Address::EDGE;
    } else if (texture.wrap_u == gobot::TextureWrap::MirroredRepeat ||
               texture.wrap_v == gobot::TextureWrap::MirroredRepeat) {
        address = Sampler::Address::MIRROR;
    }
    return Sampler{filter, address};
}

struct GeometryKey {
    std::uint64_t mesh_id = 0;
    std::uint64_t revision = 0;
    std::size_t surface = 0;
    bool operator==(const GeometryKey&) const = default;
};

struct GeometryKeyHash {
    std::size_t operator()(const GeometryKey& key) const {
        std::size_t hash = std::hash<std::uint64_t>{}(key.mesh_id);
        hash ^= std::hash<std::uint64_t>{}(key.revision) + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
        hash ^= std::hash<std::size_t>{}(key.surface) + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
        return hash;
    }
};

struct TextureKey {
    std::uint64_t texture_id = 0;
    std::uint64_t texture_revision = 0;
    std::uint64_t image_id = 0;
    std::uint64_t image_revision = 0;
    bool operator==(const TextureKey&) const = default;
};

struct TextureKeyHash {
    std::size_t operator()(const TextureKey& key) const {
        std::size_t hash = std::hash<std::uint64_t>{}(key.texture_id);
        hash ^= std::hash<std::uint64_t>{}(key.texture_revision) + (hash << 6U) + (hash >> 2U);
        hash ^= std::hash<std::uint64_t>{}(key.image_id) + (hash << 6U) + (hash >> 2U);
        hash ^= std::hash<std::uint64_t>{}(key.image_revision) + (hash << 6U) + (hash >> 2U);
        return hash;
    }
};

struct GeometryResource {
    Buffer<GpuVertex> vertices;
    Buffer<Triangle> triangles;
    Mesh mesh;
};

struct TextureResource {
    Image<float> image;
};

class LuisaRenderer {
public:
    explicit LuisaRenderer(const std::string& module_directory)
        : runtime_directory_(ResolveRuntimeDirectory(module_directory)),
          context_(std::make_unique<Context>(
                  (std::filesystem::path(runtime_directory_) / "gobot_luisa_runtime").string(),
                  module_directory)),
          binary_io_(std::make_unique<GobotBinaryIO>(runtime_directory_)),
          device_(std::make_unique<Device>(CreateCudaDevice(*context_, binary_io_.get()))),
          stream_(std::make_unique<Stream>(device_->create_stream(StreamTag::COMPUTE))) {
        CompileShaders();
    }

    ~LuisaRenderer() {
        ReleaseInterop();
        if (stream_ != nullptr) {
            *stream_ << synchronize();
        }
    }

    gobot::SceneRendererCapabilities Capabilities() const {
        return MakeCapabilities();
    }

    void ResetAccumulation() {
        accumulation_valid_ = false;
        accumulated_samples_ = 0;
    }

    gobot::LuisaRendererResult Render(const gobot::LuisaRendererTarget& target,
                                      const gobot::SceneRenderSnapshot& snapshot,
                                      const gobot::SceneRendererSettings& settings,
                                      gobot::SceneRendererStats* stats,
                                      std::string* error) {
        const auto scene_start = std::chrono::steady_clock::now();
        if (!SyncScene(snapshot, error)) {
            return gobot::LuisaRendererResult::RecoverableError;
        }
        if (!EnsureFrameResources(target.width, target.height, error)) {
            return gobot::LuisaRendererResult::RecoverableError;
        }
        const auto scene_end = std::chrono::steady_clock::now();

        const bool scene_changed = last_combined_ != snapshot.fingerprints.combined;
        if (scene_changed) {
            ResetAccumulation();
            stable_frame_count_ = 0;
        } else {
            ++stable_frame_count_;
        }
        last_combined_ = snapshot.fingerprints.combined;

        gobot::SceneRendererMode active_mode = settings.mode;
        if (active_mode == gobot::SceneRendererMode::RayTracingAuto) {
            active_mode = stable_frame_count_ < 2
                                  ? gobot::SceneRendererMode::RealtimeRayTracing
                                  : gobot::SceneRendererMode::ProgressivePathTracing;
        }
        if (active_mode == gobot::SceneRendererMode::RealtimeRayTracing) {
            ResetAccumulation();
        }
        if (!accumulation_valid_) {
            const uint2 resolution = make_uint2(static_cast<uint>(target.width), static_cast<uint>(target.height));
            *stream_ << clear_shader_(accumulation_, guide_).dispatch(resolution)
                     << seed_shader_(seeds_, static_cast<uint>(snapshot.fingerprints.combined)).dispatch(resolution);
            accumulation_valid_ = true;
        }

        int sample_count = active_mode == gobot::SceneRendererMode::RealtimeRayTracing
                                   ? 1
                                   : std::max(1, settings.samples_per_frame);
        if (settings.adaptive_quality && previous_render_ms_ > 0.0) {
            const double budget_ms = 1000.0 / std::max(1, settings.target_fps);
            sample_count = std::max(1, static_cast<int>(std::floor(
                                              sample_count * budget_ms / previous_render_ms_)));
            sample_count = std::min(sample_count, std::max(1, settings.samples_per_frame));
        }
        const std::uint64_t remaining = accumulated_samples_ <
                                                static_cast<std::uint64_t>(settings.max_accumulated_samples)
                                                ? static_cast<std::uint64_t>(settings.max_accumulated_samples) -
                                                          accumulated_samples_
                                                : 0;
        sample_count = static_cast<int>(std::min<std::uint64_t>(sample_count, remaining));

        const uint2 resolution = make_uint2(static_cast<uint>(target.width), static_cast<uint>(target.height));
        const gobot::Matrix4 inverse_view_projection = snapshot.camera.view_projection.inverse();
        const auto render_start = std::chrono::steady_clock::now();
        for (int sample = 0; sample < sample_count; ++sample) {
            *stream_ << trace_shader_(accumulation_,
                                      seeds_,
                                      guide_,
                                      accel_,
                                      geometry_heap_,
                                      texture_heap_,
                                      materials_,
                                      lights_,
                                      static_cast<uint>(active_light_count_),
                                      environment_texture_slot_,
                                      make_float3(snapshot.environment.sky_color.red(),
                                                  snapshot.environment.sky_color.green(),
                                                  snapshot.environment.sky_color.blue()),
                                      make_float3(snapshot.environment.ground_color.red(),
                                                  snapshot.environment.ground_color.green(),
                                                  snapshot.environment.ground_color.blue()),
                                      static_cast<float>(snapshot.environment.environment_intensity),
                                      ToLuisaMatrix(inverse_view_projection),
                                      ToFloat3(snapshot.camera.world_position),
                                      static_cast<uint>(settings.max_bounces),
                                      static_cast<uint>(accumulated_samples_ + sample))
                                .dispatch(resolution);
        }
        accumulated_samples_ += sample_count;
        *stream_ << tone_shader_(accumulation_,
                                 guide_,
                                 presentation_,
                                 static_cast<float>(snapshot.environment.exposure),
                                 settings.denoise ? 1u : 0u,
                                 resolution)
                            .dispatch(resolution)
                 << synchronize();
        const auto render_end = std::chrono::steady_clock::now();
        previous_render_ms_ = std::chrono::duration<double, std::milli>(render_end - render_start).count();

        const auto presentation_start = std::chrono::steady_clock::now();
        if (!Present(target, error)) {
            return gobot::LuisaRendererResult::RecoverableError;
        }
        const auto presentation_end = std::chrono::steady_clock::now();

        if (stats != nullptr) {
            stats->active_mode = active_mode;
            stats->accumulated_samples = accumulated_samples_;
            stats->scene_update_ms = std::chrono::duration<double, std::milli>(scene_end - scene_start).count();
            stats->render_ms = previous_render_ms_;
            stats->denoise_ms = settings.denoise ? previous_render_ms_ : 0.0;
            stats->presentation_ms =
                    std::chrono::duration<double, std::milli>(presentation_end - presentation_start).count();
            stats->status = active_mode == gobot::SceneRendererMode::RealtimeRayTracing
                                    ? "LuisaCompute realtime"
                                    : "LuisaCompute progressive";
        }
        return gobot::LuisaRendererResult::Success;
    }

private:
    static std::string ResolveRuntimeDirectory(const std::string& module_directory) {
        const std::filesystem::path packaged_runtime =
                std::filesystem::path(module_directory) / "luisa";
        return std::filesystem::is_directory(packaged_runtime)
                       ? packaged_runtime.string()
                       : module_directory;
    }

    using TraceShader = Shader2D<Image<float>,
                                 Image<uint>,
                                 Image<float>,
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
    using ClearShader = Shader2D<Image<float>, Image<float>>;
    using SeedShader = Shader2D<Image<uint>, uint>;
    using ToneShader = Shader2D<Image<float>, Image<float>, Buffer<uint>, float, uint, uint2>;

    std::string runtime_directory_;
    std::unique_ptr<Context> context_;
    std::unique_ptr<GobotBinaryIO> binary_io_;
    std::unique_ptr<Device> device_;
    std::unique_ptr<Stream> stream_;
    TraceShader trace_shader_;
    ClearShader clear_shader_;
    SeedShader seed_shader_;
    ToneShader tone_shader_;

    std::unordered_map<GeometryKey, std::unique_ptr<GeometryResource>, GeometryKeyHash> geometry_cache_;
    std::unordered_map<TextureKey, std::unique_ptr<TextureResource>, TextureKeyHash> texture_cache_;
    std::vector<GeometryResource*> active_geometry_;
    Accel accel_;
    BindlessArray geometry_heap_;
    BindlessArray texture_heap_;
    Buffer<GpuMaterial> materials_;
    Buffer<GpuLight> lights_;
    std::size_t active_light_count_ = 0;
    std::uint32_t next_texture_slot_ = 0;
    std::uint32_t environment_texture_slot_ = kInvalidTexture;

    Image<float> accumulation_;
    Image<float> guide_;
    Image<uint> seeds_;
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

    CUgraphicsResource interop_resource_ = nullptr;
    std::uint32_t interop_texture_ = 0;

    void CompileShaders() {
        Callable tea = [](UInt value0, UInt value1) noexcept {
            UInt sum = def(0u);
            $for (i, 4u) {
                sum += 0x9e3779b9u;
                value0 += ((value1 << 4u) + 0xa341316cu) ^ (value1 + sum) ^
                          ((value1 >> 5u) + 0xc8013ea4u);
                value1 += ((value0 << 4u) + 0xad90777du) ^ (value0 + sum) ^
                          ((value0 >> 5u) + 0x7e95761eu);
            };
            return value0;
        };
        Callable random_float = [](UInt& state) noexcept {
            state = state * 1664525u + 1013904223u;
            return cast<float>(state & 0x00ffffffu) * (1.0f / 16777216.0f);
        };
        Callable make_onb = [](Float3 normal) noexcept {
            Float3 binormal = normalize(ite(abs(normal.x) > abs(normal.z),
                                            make_float3(-normal.y, normal.x, 0.0f),
                                            make_float3(0.0f, -normal.z, normal.y)));
            return def<Onb>(normalize(cross(binormal, normal)), binormal, normal);
        };
        Callable cosine_sample = [](Float2 sample) noexcept {
            Float radius = sqrt(sample.x);
            Float phi = 2.0f * constants::pi * sample.y;
            return make_float3(radius * cos(phi), radius * sin(phi), sqrt(1.0f - sample.x));
        };
        Callable fresnel = [](Float cosine, Float3 f0) noexcept {
            return f0 + (1.0f - f0) * pow(clamp(1.0f - cosine, 0.0f, 1.0f), 5.0f);
        };
        Callable direct_brdf = [&](Float3 normal,
                                   Float3 view,
                                   Float3 light,
                                   Float3 radiance,
                                   Float3 albedo,
                                   Float metallic,
                                   Float roughness,
                                   Float specular_weight) noexcept {
            Float3 half_vector = normalize(view + light);
            Float n_dot_l = max(dot(normal, light), 0.0f);
            Float n_dot_v = max(dot(normal, view), 0.0f);
            Float n_dot_h = max(dot(normal, half_vector), 0.0f);
            Float v_dot_h = max(dot(view, half_vector), 0.0f);
            Float alpha = roughness * roughness;
            Float alpha2 = alpha * alpha;
            Float denominator = n_dot_h * n_dot_h * (alpha2 - 1.0f) + 1.0f;
            Float distribution = alpha2 / max(constants::pi * denominator * denominator, 1.0e-4f);
            Float k = (roughness + 1.0f) * (roughness + 1.0f) * 0.125f;
            Float geometry_v = n_dot_v / max(n_dot_v * (1.0f - k) + k, 1.0e-4f);
            Float geometry_l = n_dot_l / max(n_dot_l * (1.0f - k) + k, 1.0e-4f);
            Float3 f0 = lerp(make_float3(0.02f + 0.06f * specular_weight), albedo, metallic);
            Float3 f = fresnel(v_dot_h, f0);
            Float3 specular = distribution * geometry_v * geometry_l * f /
                              max(4.0f * n_dot_v * n_dot_l, 1.0e-4f);
            Float3 diffuse = (1.0f - f) * (1.0f - metallic) * albedo * constants::inv_pi;
            return (diffuse + specular) * radiance * n_dot_l;
        };

        Kernel2D clear_kernel = [](ImageFloat accumulation, ImageFloat guide) noexcept {
            UInt2 pixel = dispatch_id().xy();
            accumulation.write(pixel, make_float4(0.0f));
            guide.write(pixel, make_float4(0.0f));
        };
        Kernel2D seed_kernel = [tea](ImageUInt seeds, UInt frame_seed) noexcept {
            UInt2 pixel = dispatch_id().xy();
            seeds.write(pixel, make_uint4(tea(pixel.x ^ frame_seed, pixel.y), 0u, 0u, 0u));
        };

        Kernel2D trace_kernel = [=](ImageFloat accumulation,
                                    ImageUInt seeds,
                                    ImageFloat guide,
                                    AccelVar accel,
                                    BindlessVar geometry,
                                    BindlessVar textures,
                                    BufferVar<GpuMaterial> materials,
                                    BufferVar<GpuLight> lights,
                                    UInt light_count,
                                    UInt environment_texture,
                                    Float3 sky_color,
                                    Float3 ground_color,
                                    Float environment_intensity,
                                    Float4x4 inverse_view_projection,
                                    Float3 camera_position,
                                    UInt max_bounces,
                                    UInt frame_index) noexcept {
            set_block_size(16u, 16u, 1u);
            UInt2 pixel = dispatch_id().xy();
            Float2 resolution = make_float2(dispatch_size().xy());
            UInt state = seeds.read(pixel).x ^ tea(frame_index, pixel.x + pixel.y * dispatch_size().x);
            Float2 jitter = make_float2(random_float(state), random_float(state));
            Float2 ndc = (make_float2(pixel) + jitter) / resolution * 2.0f - 1.0f;
            Float4 near_h = inverse_view_projection * make_float4(ndc, -1.0f, 1.0f);
            Float4 far_h = inverse_view_projection * make_float4(ndc, 1.0f, 1.0f);
            Float3 near_point = near_h.xyz() / near_h.w;
            Float3 far_point = far_h.xyz() / far_h.w;
            Var<Ray> ray = make_ray(camera_position, normalize(far_point - near_point));
            Float3 throughput = def(make_float3(1.0f));
            Float3 radiance = def(make_float3(0.0f));
            Bool wrote_guide = def(false);

            $for (depth, 12u) {
                $if (depth >= max_bounces) { $break; };
                Var<TriangleHit> hit = accel.intersect(ray, {});
                $if (hit->miss()) {
                    Float3 direction = ray->direction();
                    Float sky_mix = clamp(direction.z * 0.5f + 0.5f, 0.0f, 1.0f);
                    Float3 environment = lerp(ground_color, sky_color, sky_mix);
                    $if (environment_texture != kInvalidTexture) {
                        Float2 uv = make_float2(
                                atan2(direction.y, direction.x) / (2.0f * constants::pi) + 0.5f,
                                acos(clamp(direction.z, -1.0f, 1.0f)) / constants::pi);
                        environment = textures.tex2d(environment_texture).sample(uv).xyz() *
                                      environment_intensity;
                    };
                    radiance += throughput * environment;
                    $break;
                };

                Var<Triangle> triangle = geometry.buffer<Triangle>(hit.inst * 2u + 1u).read(hit.prim);
                Var<GpuVertex> vertex0 = geometry.buffer<GpuVertex>(hit.inst * 2u).read(triangle.i0);
                Var<GpuVertex> vertex1 = geometry.buffer<GpuVertex>(hit.inst * 2u).read(triangle.i1);
                Var<GpuVertex> vertex2 = geometry.buffer<GpuVertex>(hit.inst * 2u).read(triangle.i2);
                Float4x4 model = accel.instance_transform(hit.inst);
                Float4x4 normal_transform = transpose(inverse(model));
                Float3 local_position = triangle_interpolate(
                        hit.bary, vertex0.position, vertex1.position, vertex2.position);
                Float3 position = (model * make_float4(local_position, 1.0f)).xyz();
                Float3 normal = normalize((normal_transform * make_float4(
                        triangle_interpolate(hit.bary, vertex0.normal, vertex1.normal, vertex2.normal),
                        0.0f)).xyz());
                normal = faceforward(normal, ray->direction(), normal);
                Float4 tangent4 = triangle_interpolate(
                        hit.bary, vertex0.tangent, vertex1.tangent, vertex2.tangent);
                Float3 tangent = normalize((model * make_float4(tangent4.xyz(), 0.0f)).xyz());
                tangent = normalize(tangent - normal * dot(normal, tangent));
                Float3 bitangent = normalize(cross(normal, tangent)) * tangent4.w;
                Float2 uv = triangle_interpolate(hit.bary, vertex0.uv, vertex1.uv, vertex2.uv);
                Float4 vertex_color = triangle_interpolate(
                        hit.bary, vertex0.color, vertex1.color, vertex2.color);
                Var<GpuMaterial> material = materials.read(hit.inst);
                Float4 albedo = material.albedo * vertex_color;
                $if (material.textures0.x != kInvalidTexture) {
                    Float4 sampled = textures.tex2d(material.textures0.x).sample(uv);
                    albedo *= make_float4(pow(max(sampled.xyz(), make_float3(0.0f)), 2.2f), sampled.w);
                };
                Float metallic = material.pbr.x;
                Float roughness = clamp(material.pbr.y, 0.04f, 1.0f);
                $if (material.textures0.y != kInvalidTexture) {
                    Float4 sampled = textures.tex2d(material.textures0.y).sample(uv);
                    roughness = clamp(roughness * sampled.y, 0.04f, 1.0f);
                    metallic = clamp(metallic * sampled.z, 0.0f, 1.0f);
                };
                $if (material.textures0.z != kInvalidTexture) {
                    Float3 mapped = textures.tex2d(material.textures0.z).sample(uv).xyz() * 2.0f - 1.0f;
                    mapped = make_float3(mapped.xy() * material.pbr.w, mapped.z);
                    normal = normalize(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z);
                };
                Float3 emission = material.emissive.xyz();
                $if (material.textures1.x != kInvalidTexture) {
                    Float3 sampled = textures.tex2d(material.textures1.x).sample(uv).xyz();
                    emission *= pow(max(sampled, make_float3(0.0f)), 2.2f);
                };
                radiance += throughput * emission;

                $if (!wrote_guide) {
                    guide.write(pixel, make_float4(normal, length(position - camera_position)));
                    wrote_guide = true;
                };

                Float3 view_direction = -ray->direction();
                $for (light_index, kMaxLights) {
                    $if (light_index < light_count) {
                        Var<GpuLight> light = lights.read(light_index);
                        UInt light_type = cast<uint>(light.position_type.w);
                        Float3 light_direction = light.direction_range.xyz();
                        Float max_distance = def(1.0e20f);
                        Float attenuation = def(1.0f);
                        $if (light_type != 0u) {
                            Float3 to_light = light.position_type.xyz() - position;
                            max_distance = length(to_light);
                            light_direction = to_light / max(max_distance, 1.0e-4f);
                            Float range_weight = clamp(
                                    1.0f - max_distance / max(light.direction_range.w, 1.0e-3f),
                                    0.0f,
                                    1.0f);
                            attenuation = range_weight * range_weight /
                                          max(max_distance * max_distance, 0.01f);
                            $if (light_type == 2u) {
                                Float cone = dot(-light_direction, normalize(light.direction_range.xyz()));
                                attenuation *= smoothstep(light.spot_cosines.y,
                                                          light.spot_cosines.x,
                                                          cone);
                            };
                        };
                        Var<Ray> shadow_ray = make_ray(offset_ray_origin(position, normal),
                                                       light_direction,
                                                       0.0f,
                                                       max_distance - 1.0e-3f);
                        Bool occluded = accel.intersect_any(shadow_ray, {});
                        $if (!occluded) {
                            Float3 light_radiance = light.color_intensity.xyz() *
                                                   light.color_intensity.w * attenuation;
                            radiance += throughput * direct_brdf(normal,
                                                                 view_direction,
                                                                 light_direction,
                                                                 light_radiance,
                                                                 albedo.xyz(),
                                                                 metallic,
                                                                 roughness,
                                                                 material.pbr.z);
                        };
                    };
                };

                Var<Onb> basis = make_onb(normal);
                Float3 diffuse_direction = basis->to_world(cosine_sample(
                        make_float2(random_float(state), random_float(state))));
                Float3 reflected = reflect(ray->direction(), normal);
                Float3 glossy_direction = normalize(lerp(
                        reflected,
                        basis->to_world(cosine_sample(make_float2(
                                random_float(state), random_float(state)))),
                        roughness * roughness));
                Bool choose_metal = random_float(state) < metallic;
                Float3 next_direction = ite(choose_metal, glossy_direction, diffuse_direction);
                Float3 f0 = lerp(make_float3(0.02f + 0.06f * material.pbr.z), albedo.xyz(), metallic);
                throughput *= ite(choose_metal, fresnel(max(dot(normal, next_direction), 0.0f), f0), albedo.xyz());
                ray = make_ray(offset_ray_origin(position, normal), next_direction);

                $if (depth >= 2u) {
                    Float survival = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05f, 0.98f);
                    $if (random_float(state) > survival) { $break; };
                    throughput /= survival;
                };
            };

            Float4 previous = accumulation.read(pixel);
            accumulation.write(pixel, previous + make_float4(max(radiance, make_float3(0.0f)), 1.0f));
            seeds.write(pixel, make_uint4(state, 0u, 0u, 0u));
        };

        Kernel2D tone_kernel = [](ImageFloat accumulation,
                                  ImageFloat guide,
                                  BufferUInt output,
                                  Float exposure,
                                  UInt denoise,
                                  UInt2 resolution) noexcept {
            UInt2 pixel = dispatch_id().xy();
            Float4 center_accum = accumulation.read(pixel);
            Float3 color = center_accum.xyz() / max(center_accum.w, 1.0f);
            $if ((denoise != 0u) & (center_accum.w < 128.0f)) {
                Float4 center_guide = guide.read(pixel);
                Float3 weighted = def(make_float3(0.0f));
                Float total_weight = def(0.0f);
                $for (offset_y, 5u) {
                    $for (offset_x, 5u) {
                        Int2 offset = make_int2(cast<int>(offset_x) - 2, cast<int>(offset_y) - 2);
                        Int2 sample_position = clamp(make_int2(pixel) + offset,
                                                     make_int2(0),
                                                     make_int2(resolution) - 1);
                        Float4 sample_accum = accumulation.read(make_uint2(sample_position));
                        Float4 sample_guide = guide.read(make_uint2(sample_position));
                        Float normal_weight = pow(max(dot(center_guide.xyz(), sample_guide.xyz()), 0.0f), 32.0f);
                        Float depth_weight = exp(-abs(center_guide.w - sample_guide.w) /
                                                 max(0.05f, center_guide.w * 0.03f));
                        Float spatial_weight = exp(-0.35f * dot(make_float2(offset), make_float2(offset)));
                        Float weight = normal_weight * depth_weight * spatial_weight;
                        weighted += sample_accum.xyz() / max(sample_accum.w, 1.0f) * weight;
                        total_weight += weight;
                    };
                };
                color = weighted / max(total_weight, 1.0e-4f);
            };
            color *= exposure;
            color = clamp((color * (2.51f * color + 0.03f)) /
                          (color * (2.43f * color + 0.59f) + 0.14f),
                          0.0f,
                          1.0f);
            color = pow(color, 1.0f / 2.2f);
            UInt red = cast<uint>(round(color.x * 255.0f));
            UInt green = cast<uint>(round(color.y * 255.0f));
            UInt blue = cast<uint>(round(color.z * 255.0f));
            UInt packed = red | (green << 8u) | (blue << 16u) | (255u << 24u);
            output.write(pixel.y * resolution.x + pixel.x, packed);
        };

        ShaderOption option{.enable_debug_info = false};
        clear_shader_ = device_->compile(clear_kernel, option);
        seed_shader_ = device_->compile(seed_kernel, option);
        trace_shader_ = device_->compile(trace_kernel, option);
        tone_shader_ = device_->compile(tone_kernel, option);
    }

    GeometryResource* EnsureGeometry(const gobot::VisualMeshRenderItem& item, std::string* error) {
        const gobot::MeshSurfaceData* surface = item.GetSurface();
        if (surface == nullptr) {
            *error = "Render snapshot contains an invalid mesh surface.";
            return nullptr;
        }
        const GeometryKey key{
                item.mesh_id.operator std::uint64_t(), item.mesh_revision, item.surface_index};
        if (const auto found = geometry_cache_.find(key); found != geometry_cache_.end()) {
            return found->second.get();
        }

        std::vector<GpuVertex> vertices;
        vertices.reserve(surface->vertices.size());
        for (std::size_t i = 0; i < surface->vertices.size(); ++i) {
            const gobot::Vector3 normal = surface->normals.size() == surface->vertices.size()
                                                  ? surface->normals[i]
                                                  : gobot::Vector3::UnitZ();
            const gobot::Vector4 tangent = surface->tangents.size() == surface->vertices.size()
                                                   ? surface->tangents[i]
                                                   : gobot::Vector4{1.0, 0.0, 0.0, 1.0};
            const gobot::Vector2 uv = surface->uv0.size() == surface->vertices.size()
                                              ? surface->uv0[i]
                                              : gobot::Vector2::Zero();
            const gobot::Color color = surface->colors.size() == surface->vertices.size()
                                                 ? surface->colors[i]
                                                 : gobot::Color{1.0f, 1.0f, 1.0f, 1.0f};
            vertices.push_back({
                    ToFloat3(surface->vertices[i]),
                    ToFloat3(normal),
                    make_float4(tangent.x(), tangent.y(), tangent.z(), tangent.w()),
                    make_float2(uv.x(), uv.y()),
                    ToFloat4(color)});
        }
        std::vector<Triangle> triangles;
        triangles.reserve(surface->indices.size() / 3);
        for (std::size_t i = 0; i + 2 < surface->indices.size(); i += 3) {
            triangles.push_back({surface->indices[i], surface->indices[i + 1], surface->indices[i + 2]});
        }
        if (vertices.empty() || triangles.empty()) {
            *error = "Render snapshot contains empty geometry.";
            return nullptr;
        }

        auto resource = std::make_unique<GeometryResource>();
        resource->vertices = device_->create_buffer<GpuVertex>(vertices.size());
        resource->triangles = device_->create_buffer<Triangle>(triangles.size());
        resource->mesh = device_->create_mesh(resource->vertices, resource->triangles);
        *stream_ << resource->vertices.copy_from(luisa::span{vertices})
                 << resource->triangles.copy_from(luisa::span{triangles})
                 << resource->mesh.build();
        GeometryResource* result = resource.get();
        geometry_cache_.emplace(key, std::move(resource));
        return result;
    }

    std::uint32_t BindTexture(const gobot::RenderTextureSnapshot& texture) {
        if (!texture.IsValid() || next_texture_slot_ >= 65535u) {
            return kInvalidTexture;
        }
        const TextureKey key{
                texture.texture_id.operator std::uint64_t(),
                texture.revision,
                texture.image.image_id.operator std::uint64_t(),
                texture.image.revision};
        TextureResource* resource = nullptr;
        if (const auto found = texture_cache_.find(key); found != texture_cache_.end()) {
            resource = found->second.get();
        } else {
            std::vector<float4> pixels = ConvertImage(*texture.image.storage);
            if (pixels.empty()) {
                return kInvalidTexture;
            }
            auto created = std::make_unique<TextureResource>();
            created->image = device_->create_image<float>(
                    PixelStorage::FLOAT4,
                    make_uint2(static_cast<uint>(texture.image.storage->width),
                               static_cast<uint>(texture.image.storage->height)));
            *stream_ << created->image.copy_from(luisa::span{pixels});
            resource = created.get();
            texture_cache_.emplace(key, std::move(created));
        }
        const std::uint32_t slot = next_texture_slot_++;
        texture_heap_.emplace_on_update(slot, resource->image, ToSampler(texture));
        return slot;
    }

    GpuMaterial MakeMaterial(const gobot::RenderMaterialSnapshot& material) {
        return {
                ToFloat4(material.albedo),
                make_float4(material.emissive.red(),
                            material.emissive.green(),
                            material.emissive.blue(),
                            0.0f),
                make_float4(material.metallic,
                            material.roughness,
                            material.specular,
                            material.normal_scale),
                make_float4(material.occlusion_strength,
                            material.alpha_cutoff,
                            static_cast<float>(material.alpha_mode),
                            material.double_sided ? 1.0f : 0.0f),
                make_uint4(BindTexture(material.albedo_texture),
                           BindTexture(material.metallic_roughness_texture),
                           BindTexture(material.normal_texture),
                           BindTexture(material.occlusion_texture)),
                make_uint4(BindTexture(material.emissive_texture),
                           kInvalidTexture,
                           kInvalidTexture,
                           kInvalidTexture)};
    }

    bool RebuildTopology(const gobot::SceneRenderSnapshot& snapshot, std::string* error) {
        active_geometry_.clear();
        accel_ = device_->create_accel({});
        geometry_heap_ = device_->create_bindless_array(
                std::max<std::size_t>(1, snapshot.visual_meshes.size() * 2));
        for (std::size_t i = 0; i < snapshot.visual_meshes.size(); ++i) {
            GeometryResource* geometry = EnsureGeometry(snapshot.visual_meshes[i], error);
            if (geometry == nullptr) {
                return false;
            }
            active_geometry_.push_back(geometry);
            geometry_heap_.emplace_on_update(i * 2, geometry->vertices);
            geometry_heap_.emplace_on_update(i * 2 + 1, geometry->triangles);
            accel_.emplace_back(geometry->mesh,
                                ToLuisaMatrix(snapshot.visual_meshes[i].model),
                                0xffu,
                                snapshot.visual_meshes[i].material.alpha_mode == gobot::AlphaMode::Opaque,
                                static_cast<uint>(i));
        }
        *stream_ << geometry_heap_.update()
                 << accel_.build(AccelBuildRequest::FORCE_BUILD)
                 << synchronize();
        return true;
    }

    void UpdateTransforms(const gobot::SceneRenderSnapshot& snapshot) {
        for (std::size_t i = 0; i < snapshot.visual_meshes.size(); ++i) {
            accel_.set_transform_on_update(i, ToLuisaMatrix(snapshot.visual_meshes[i].model));
        }
        *stream_ << accel_.build(AccelBuildRequest::PREFER_UPDATE) << synchronize();
    }

    void UpdateMaterialsAndLighting(const gobot::SceneRenderSnapshot& snapshot) {
        texture_heap_ = device_->create_bindless_array(65536);
        next_texture_slot_ = 0;
        std::vector<GpuMaterial> host_materials;
        host_materials.reserve(std::max<std::size_t>(1, snapshot.visual_meshes.size()));
        for (const gobot::VisualMeshRenderItem& item : snapshot.visual_meshes) {
            host_materials.emplace_back(MakeMaterial(item.material));
        }
        if (host_materials.empty()) {
            host_materials.emplace_back(MakeMaterial({}));
        }
        environment_texture_slot_ = BindTexture(snapshot.environment.environment_texture);

        std::vector<GpuLight> host_lights;
        host_lights.reserve(std::max<std::size_t>(1, snapshot.lights.size()));
        for (const gobot::RenderLightSnapshot& light : snapshot.lights) {
            const float inner = std::cos(static_cast<float>(light.inner_angle * M_PI / 180.0));
            const float outer = std::cos(static_cast<float>(light.outer_angle * M_PI / 180.0));
            host_lights.push_back({
                    make_float4(ToFloat3(light.position), static_cast<float>(light.type)),
                    make_float4(ToFloat3(light.direction), static_cast<float>(light.range)),
                    make_float4(light.color.red(),
                                light.color.green(),
                                light.color.blue(),
                                static_cast<float>(light.intensity)),
                    make_float4(inner, outer, 0.0f, 0.0f)});
        }
        active_light_count_ = std::min<std::size_t>(host_lights.size(), kMaxLights);
        if (host_lights.empty()) {
            host_lights.push_back({});
        }
        materials_ = device_->create_buffer<GpuMaterial>(host_materials.size());
        lights_ = device_->create_buffer<GpuLight>(host_lights.size());
        *stream_ << materials_.copy_from(luisa::span{host_materials})
                 << lights_.copy_from(luisa::span{host_lights});
        if (next_texture_slot_ != 0u) {
            *stream_ << texture_heap_.update();
        }
        *stream_ << synchronize();
    }

    bool SyncScene(const gobot::SceneRenderSnapshot& snapshot, std::string* error) {
        if (snapshot.visual_meshes.empty()) {
            *error = "Scene has no renderable mesh; using raster fallback.";
            return false;
        }
        const bool topology_changed = last_topology_ != snapshot.fingerprints.topology ||
                                      last_geometry_ != snapshot.fingerprints.geometry;
        if (topology_changed && !RebuildTopology(snapshot, error)) {
            return false;
        }
        if (!topology_changed && last_transforms_ != snapshot.fingerprints.transforms) {
            UpdateTransforms(snapshot);
        }
        if (topology_changed || last_materials_ != snapshot.fingerprints.materials ||
            last_lighting_ != snapshot.fingerprints.lighting) {
            UpdateMaterialsAndLighting(snapshot);
        }
        last_topology_ = snapshot.fingerprints.topology;
        last_geometry_ = snapshot.fingerprints.geometry;
        last_transforms_ = snapshot.fingerprints.transforms;
        last_materials_ = snapshot.fingerprints.materials;
        last_lighting_ = snapshot.fingerprints.lighting;
        return true;
    }

    bool EnsureFrameResources(int width, int height, std::string* error) {
        if (width <= 0 || height <= 0) {
            *error = "Invalid Luisa renderer output size.";
            return false;
        }
        if (frame_width_ == width && frame_height_ == height && presentation_) {
            return true;
        }
        frame_width_ = width;
        frame_height_ = height;
        const uint2 resolution = make_uint2(static_cast<uint>(width), static_cast<uint>(height));
        accumulation_ = device_->create_image<float>(PixelStorage::FLOAT4, resolution);
        guide_ = device_->create_image<float>(PixelStorage::FLOAT4, resolution);
        seeds_ = device_->create_image<uint>(PixelStorage::INT1, resolution);
        presentation_ = device_->create_buffer<uint>(static_cast<std::size_t>(width) * height);
        ResetAccumulation();
        ReleaseInterop();
        return true;
    }

    bool EnsureInterop(std::uint32_t texture, std::string* error) {
        if (interop_resource_ != nullptr && interop_texture_ == texture) {
            return true;
        }
        ReleaseInterop();
        CUcontext context = reinterpret_cast<CUcontext>(device_->native_handle());
        CUresult result = cuCtxPushCurrent(context);
        if (result != CUDA_SUCCESS) {
            *error = CudaError(result, "cuCtxPushCurrent");
            return false;
        }
        result = cuGraphicsGLRegisterImage(&interop_resource_,
                                           texture,
                                           GL_TEXTURE_2D,
                                           CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
        CUcontext popped = nullptr;
        cuCtxPopCurrent(&popped);
        if (result != CUDA_SUCCESS) {
            interop_resource_ = nullptr;
            *error = CudaError(result, "cuGraphicsGLRegisterImage");
            return false;
        }
        interop_texture_ = texture;
        return true;
    }

    bool Present(const gobot::LuisaRendererTarget& target, std::string* error) {
        if (!EnsureInterop(target.gl_color_texture, error)) {
            return false;
        }
        CUcontext context = reinterpret_cast<CUcontext>(device_->native_handle());
        CUstream stream = reinterpret_cast<CUstream>(stream_->native_handle());
        CUresult result = cuCtxPushCurrent(context);
        if (result != CUDA_SUCCESS) {
            *error = CudaError(result, "cuCtxPushCurrent");
            return false;
        }
        result = cuGraphicsMapResources(1, &interop_resource_, stream);
        CUarray destination = nullptr;
        if (result == CUDA_SUCCESS) {
            result = cuGraphicsSubResourceGetMappedArray(&destination, interop_resource_, 0, 0);
        }
        if (result == CUDA_SUCCESS) {
            CUDA_MEMCPY2D copy{};
            copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copy.srcDevice = reinterpret_cast<CUdeviceptr>(presentation_.native_handle());
            copy.srcPitch = static_cast<std::size_t>(target.width) * sizeof(std::uint32_t);
            copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            copy.dstArray = destination;
            copy.WidthInBytes = static_cast<std::size_t>(target.width) * sizeof(std::uint32_t);
            copy.Height = target.height;
            result = cuMemcpy2DAsync(&copy, stream);
        }
        if (result == CUDA_SUCCESS) {
            result = cuGraphicsUnmapResources(1, &interop_resource_, stream);
        }
        if (result == CUDA_SUCCESS) {
            result = cuStreamSynchronize(stream);
        }
        CUcontext popped = nullptr;
        cuCtxPopCurrent(&popped);
        if (result != CUDA_SUCCESS) {
            *error = CudaError(result, "CUDA-OpenGL presentation");
            return false;
        }
        return true;
    }

    void ReleaseInterop() {
        if (interop_resource_ == nullptr || device_ == nullptr) {
            return;
        }
        CUcontext context = reinterpret_cast<CUcontext>(device_->native_handle());
        if (cuCtxPushCurrent(context) == CUDA_SUCCESS) {
            cuGraphicsUnregisterResource(interop_resource_);
            CUcontext popped = nullptr;
            cuCtxPopCurrent(&popped);
        }
        interop_resource_ = nullptr;
        interop_texture_ = 0;
    }
};

void* CreateRenderer(const char* module_directory, char* error, std::size_t error_size) {
    try {
        return new LuisaRenderer(module_directory != nullptr ? module_directory : ".");
    } catch (const std::exception& exception) {
        SetError(error, error_size, exception.what());
        return nullptr;
    }
}

void DestroyRenderer(void* renderer) {
    delete static_cast<LuisaRenderer*>(renderer);
}

gobot::SceneRendererCapabilities GetCapabilities(void* renderer) {
    return renderer != nullptr ? static_cast<LuisaRenderer*>(renderer)->Capabilities()
                               : MakeCapabilities();
}

gobot::LuisaRendererResult Render(void* renderer,
                                  const gobot::LuisaRendererTarget* target,
                                  const gobot::SceneRenderSnapshot* snapshot,
                                  const gobot::SceneRendererSettings* settings,
                                  gobot::SceneRendererStats* stats,
                                  char* error,
                                  std::size_t error_size) {
    if (renderer == nullptr || target == nullptr || snapshot == nullptr || settings == nullptr) {
        SetError(error, error_size, "Invalid Luisa renderer call arguments.");
        return gobot::LuisaRendererResult::FatalError;
    }
    try {
        std::string message;
        const auto result = static_cast<LuisaRenderer*>(renderer)->Render(
                *target, *snapshot, *settings, stats, &message);
        if (result != gobot::LuisaRendererResult::Success) {
            SetError(error, error_size, message);
        }
        return result;
    } catch (const std::exception& exception) {
        SetError(error, error_size, exception.what());
        return gobot::LuisaRendererResult::FatalError;
    }
}

void ResetAccumulation(void* renderer) {
    if (renderer != nullptr) {
        static_cast<LuisaRenderer*>(renderer)->ResetAccumulation();
    }
}

const gobot::LuisaRendererModuleApi kApi{
        gobot::GOBOT_LUISA_RENDERER_ABI_VERSION,
        &CreateRenderer,
        &DestroyRenderer,
        &GetCapabilities,
        &Render,
        &ResetAccumulation};

} // namespace

extern "C" __attribute__((visibility("default")))
const gobot::LuisaRendererModuleApi* gobot_luisa_renderer_get_api() {
    return &kApi;
}
