/*
 * Backend-neutral camera render products and frame buffers.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gobot/core/object_id.hpp"
#include "gobot/core/rid.hpp"
#include "gobot/rendering/scene_render_items.hpp"

#include "gobot_export.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gobot {

class Camera3D;
class Node;
class PythonRenderBufferBridge;

enum class RenderOutputType : std::uint32_t {
    Rgb = 0,
    LinearDepth = 1,
    WorldNormal = 2,
    InstanceId = 3,
    SemanticId = 4
};

enum class RenderDevice {
    Auto,
    Cpu,
    Cuda
};

enum class RenderProductMode {
    Minimal,
    Realtime,
    Progressive
};

enum class RenderDataType {
    UInt8,
    Float32,
    UInt32
};

GOBOT_EXPORT const char* RenderOutputTypeName(RenderOutputType output);
GOBOT_EXPORT std::uint32_t RenderOutputBit(RenderOutputType output);
GOBOT_EXPORT bool RenderOutputMaskContains(std::uint32_t mask, RenderOutputType output);
GOBOT_EXPORT RenderDataType RenderOutputDataType(RenderOutputType output);
GOBOT_EXPORT std::size_t RenderOutputChannelCount(RenderOutputType output);
GOBOT_EXPORT std::size_t RenderDataTypeSize(RenderDataType type);

struct RenderProductDesc {
    int width = 640;
    int height = 480;
    std::vector<RenderOutputType> outputs{RenderOutputType::Rgb};
    RenderDevice device = RenderDevice::Auto;
    RenderProductMode mode = RenderProductMode::Minimal;
    std::size_t frame_slots = 3;
};

class GOBOT_EXPORT RenderBuffer {
public:
    [[nodiscard]] RenderOutputType GetOutputType() const;
    [[nodiscard]] RenderDataType GetDataType() const;
    [[nodiscard]] RenderDevice GetDevice() const;
    [[nodiscard]] int GetWidth() const;
    [[nodiscard]] int GetHeight() const;
    [[nodiscard]] std::size_t GetChannelCount() const;
    [[nodiscard]] std::size_t GetByteSize() const;
    [[nodiscard]] const void* GetData() const;
    [[nodiscard]] void* GetData();
    [[nodiscard]] std::vector<std::size_t> GetShape() const;
    bool CopyToHost(void* destination, std::size_t destination_size) const;

private:
    struct Storage {
        std::vector<std::byte> bytes;
        void* device_pointer = nullptr;
        std::size_t allocation_size = 0;
        std::size_t pixel_stride_bytes = 0;
        int device_id = 0;
        std::shared_ptr<void> backend_owner;
        std::function<bool(void*, std::size_t)> copy_to_host;
    };

    RenderBuffer(RenderOutputType output,
                 RenderDevice device,
                 int width,
                 int height,
                 std::shared_ptr<Storage> storage,
                 std::shared_ptr<void> frame_lease);

    RenderOutputType output_ = RenderOutputType::Rgb;
    RenderDevice device_ = RenderDevice::Cpu;
    int width_ = 0;
    int height_ = 0;
    std::shared_ptr<Storage> storage_;
    std::shared_ptr<void> frame_lease_;

    friend class RenderProduct;
    friend class PythonRenderBufferBridge;
};

class GOBOT_EXPORT RenderFrame {
public:
    [[nodiscard]] std::shared_ptr<RenderBuffer> Get(RenderOutputType output) const;
    [[nodiscard]] bool Contains(RenderOutputType output) const;
    [[nodiscard]] std::vector<RenderOutputType> GetOutputs() const;
    [[nodiscard]] std::uint64_t GetFrameIndex() const;
    [[nodiscard]] const std::map<std::uint32_t, std::string>& GetInstancePaths() const;
    [[nodiscard]] const std::map<std::uint32_t, std::string>& GetSemanticLabels() const;

private:
    std::map<RenderOutputType, std::shared_ptr<RenderBuffer>> buffers_;
    std::map<std::uint32_t, std::string> instance_paths_;
    std::map<std::uint32_t, std::string> semantic_labels_;
    std::shared_ptr<void> frame_lease_;
    std::uint64_t frame_index_ = 0;

    friend class RenderProduct;
};

class GOBOT_EXPORT RenderProduct {
public:
    explicit RenderProduct(RenderProductDesc desc);
    ~RenderProduct();

    RenderProduct(const RenderProduct&) = delete;
    RenderProduct& operator=(const RenderProduct&) = delete;

    [[nodiscard]] const RenderProductDesc& GetDesc() const;
    [[nodiscard]] RenderDevice GetDevice() const;
    [[nodiscard]] std::shared_ptr<RenderFrame> Capture(const RenderSceneSnapshot& scene,
                                                       const RenderViewSnapshot& view);

private:
    struct FrameSlot {
        std::map<RenderOutputType, std::shared_ptr<RenderBuffer::Storage>> storage;
        std::weak_ptr<void> lease;
    };

    void EnsureDevice();
    void EnsureCpuViewport();

    RenderProductDesc desc_;
    RenderDevice device_ = RenderDevice::Cpu;
    RID viewport_;
    std::vector<FrameSlot> slots_;
    std::uint64_t frame_index_ = 0;
    bool device_initialized_ = false;
};

class GOBOT_EXPORT CameraSensor {
public:
    CameraSensor(Camera3D* camera, Node* root, RenderProductDesc desc);

    [[nodiscard]] std::shared_ptr<RenderFrame> Capture();
    [[nodiscard]] std::shared_ptr<RenderProduct> GetRenderProduct() const;

private:
    ObjectID camera_id_;
    ObjectID root_id_;
    std::shared_ptr<RenderProduct> render_product_;
};

} // namespace gobot
