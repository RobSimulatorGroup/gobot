/*
 * Optional LuisaCompute CUDA path tracer for Gobot's OpenGL editor viewport.
 * This module is intentionally outside src/gobot so default builds never link it.
 */

#include "luisa_renderer_internal.hpp"

#if defined(GOBOT_HAS_GSPLAT_INFERENCE)
#include <gsplat_inference/renderer.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gobot::luisa_renderer {

gobot::SceneRendererCapabilities MakeCapabilities() {
    gobot::SceneRendererCapabilities capabilities;
    capabilities.ray_tracing_available = true;
    capabilities.realtime = true;
    capabilities.progressive = true;
    capabilities.denoise = true;
    capabilities.direct_presentation_interop = true;
    capabilities.cuda_render_products = true;
#if defined(GOBOT_HAS_GSPLAT_INFERENCE)
    capabilities.gaussian_splatting = true;
    capabilities.gaussian_splat_cuda = true;
#endif
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
        return base / "gobot" / "luisa" / "v2";
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

std::array<float, 16> ToFloatMatrix(const gobot::Matrix4& matrix) {
    std::array<float, 16> result{};
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            result[static_cast<std::size_t>(column * 4 + row)] =
                    static_cast<float>(matrix(row, column));
        }
    }
    return result;
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

std::size_t GeometryKeyHash::operator()(const GeometryKey& key) const {
    std::size_t hash = std::hash<std::uint64_t>{}(key.mesh_id);
    hash ^= std::hash<std::uint64_t>{}(key.revision) + 0x9e3779b9U + (hash << 6U) +
            (hash >> 2U);
    hash ^= std::hash<std::size_t>{}(key.surface) + 0x9e3779b9U + (hash << 6U) +
            (hash >> 2U);
    return hash;
}

std::size_t TextureKeyHash::operator()(const TextureKey& key) const {
    std::size_t hash = std::hash<std::uint64_t>{}(key.texture_id);
    hash ^= std::hash<std::uint64_t>{}(key.texture_revision) + (hash << 6U) + (hash >> 2U);
    hash ^= std::hash<std::uint64_t>{}(key.image_id) + (hash << 6U) + (hash >> 2U);
    hash ^= std::hash<std::uint64_t>{}(key.image_revision) + (hash << 6U) + (hash >> 2U);
    return hash;
}

std::string LuisaRenderer::ResolveRuntimeDirectory(const std::string& module_directory) {
    const std::filesystem::path packaged_runtime =
            std::filesystem::path(module_directory) / "luisa";
    return std::filesystem::is_directory(packaged_runtime) ? packaged_runtime.string()
                                                           : module_directory;
}

LuisaRenderer::LuisaRenderer(const std::string& module_directory)
        : runtime_directory_(ResolveRuntimeDirectory(module_directory)),
          context_(std::make_unique<Context>(
                  (std::filesystem::path(runtime_directory_) / "gobot_luisa_runtime").string(),
                  module_directory)),
          binary_io_(std::make_unique<GobotBinaryIO>(runtime_directory_)),
          device_(std::make_unique<Device>(CreateCudaDevice(*context_, binary_io_.get()))),
          stream_(std::make_unique<Stream>(device_->create_stream(StreamTag::COMPUTE))) {
    const CUcontext cuda_context = NativeContext();
    interop_.SetContext(cuda_context);
    render_start_event_.SetContext(cuda_context);
    render_end_event_.SetContext(cuda_context);
    presentation_end_event_.SetContext(cuda_context);
    gaussian_start_event_.SetContext(cuda_context);
    gaussian_end_event_.SetContext(cuda_context);
#if defined(GOBOT_HAS_GSPLAT_INFERENCE)
    gaussian_renderer_ = GaussianRendererPtr{
            gsplat_inference::Create(), GaussianRendererDeleter{cuda_context}};
    if (gaussian_renderer_ == nullptr) {
        throw std::runtime_error("Failed to create the gsplat inference renderer.");
    }
#endif
    CompileShaders();
    dummy_uint_ = device_->create_buffer<uint>(1);
    dummy_float_ = device_->create_buffer<float>(1);
    dummy_float4_ = device_->create_buffer<float4>(1);
}

LuisaRenderer::~LuisaRenderer() {
    if (stream_ != nullptr) {
        *stream_ << synchronize();
    }
    interop_.Reset();
#if defined(GOBOT_HAS_GSPLAT_INFERENCE)
    gaussian_renderer_.reset();
#endif
}

gobot::SceneRendererCapabilities LuisaRenderer::Capabilities() const {
    return MakeCapabilities();
}

void LuisaRenderer::ResetAccumulation() {
    accumulation_valid_ = false;
    accumulated_samples_ = 0;
}

gobot::LuisaRendererResult LuisaRenderer::Render(
        const gobot::LuisaRendererTarget& target,
        const gobot::RenderSceneSnapshot& snapshot,
        const gobot::RenderViewSnapshot& view,
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

        const std::uint64_t frame_fingerprint = snapshot.fingerprints.combined ^
                                                (view.fingerprint + 0x9e3779b97f4a7c15ULL +
                                                 (snapshot.fingerprints.combined << 6U) +
                                                 (snapshot.fingerprints.combined >> 2U));
        const bool scene_changed = last_combined_ != frame_fingerprint;
        if (scene_changed) {
            ResetAccumulation();
            stable_frame_count_ = 0;
        } else {
            ++stable_frame_count_;
        }
        last_combined_ = frame_fingerprint;

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
                     << seed_shader_(seeds_, static_cast<uint>(frame_fingerprint)).dispatch(resolution);
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
        const gobot::Matrix4 inverse_view_projection = view.camera.view_projection.inverse();
        if (!render_start_event_.Record(NativeStream(), error)) {
            return gobot::LuisaRendererResult::RecoverableError;
        }
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
                                      ToFloat3(view.camera.world_position),
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
                            .dispatch(resolution);
        if (!render_end_event_.Record(NativeStream(), error)) {
            return gobot::LuisaRendererResult::RecoverableError;
        }

        if (!Present(target, error)) {
            return gobot::LuisaRendererResult::RecoverableError;
        }
        double presentation_ms = 0.0;
        if (!render_start_event_.ElapsedMilliseconds(
                    render_end_event_, &previous_render_ms_, error) ||
            !render_end_event_.ElapsedMilliseconds(
                    presentation_end_event_, &presentation_ms, error)) {
            return gobot::LuisaRendererResult::RecoverableError;
        }

        if (stats != nullptr) {
            stats->active_mode = active_mode;
            stats->accumulated_samples = accumulated_samples_;
            stats->scene_update_ms = std::chrono::duration<double, std::milli>(scene_end - scene_start).count();
            stats->render_ms = previous_render_ms_;
            stats->denoise_ms = settings.denoise ? previous_render_ms_ : 0.0;
            stats->presentation_ms = presentation_ms;
            stats->status = active_mode == gobot::SceneRendererMode::RealtimeRayTracing
                                    ? "LuisaCompute realtime"
                                    : "LuisaCompute progressive";
            if (!snapshot.gaussian_splats.empty()) {
                stats->status += " (Gaussian background requires Raster mode)";
            }
        }
        return gobot::LuisaRendererResult::Success;
}

} // namespace gobot::luisa_renderer
