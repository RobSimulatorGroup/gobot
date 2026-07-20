#include "gobot/rendering/render_product.hpp"

#include "gobot/core/object.hpp"
#include "gobot/rendering/render_server.hpp"
#include "gobot/scene/camera_3d.hpp"
#include "gobot/scene/node.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_set>

namespace gobot {

const char* RenderOutputTypeName(RenderOutputType output) {
    switch (output) {
        case RenderOutputType::Rgb: return "rgb";
        case RenderOutputType::LinearDepth: return "linear_depth";
        case RenderOutputType::WorldNormal: return "world_normal";
        case RenderOutputType::InstanceId: return "instance_id";
        case RenderOutputType::SemanticId: return "semantic_id";
    }
    return "unknown";
}

std::uint32_t RenderOutputBit(RenderOutputType output) {
    return 1u << static_cast<std::uint32_t>(output);
}

bool RenderOutputMaskContains(std::uint32_t mask, RenderOutputType output) {
    return (mask & RenderOutputBit(output)) != 0u;
}

RenderDataType RenderOutputDataType(RenderOutputType output) {
    switch (output) {
        case RenderOutputType::Rgb: return RenderDataType::UInt8;
        case RenderOutputType::LinearDepth:
        case RenderOutputType::WorldNormal: return RenderDataType::Float32;
        case RenderOutputType::InstanceId:
        case RenderOutputType::SemanticId: return RenderDataType::UInt32;
    }
    return RenderDataType::UInt8;
}

std::size_t RenderOutputChannelCount(RenderOutputType output) {
    return output == RenderOutputType::Rgb || output == RenderOutputType::WorldNormal ? 3u : 1u;
}

std::size_t RenderDataTypeSize(RenderDataType type) {
    switch (type) {
        case RenderDataType::UInt8: return sizeof(std::uint8_t);
        case RenderDataType::Float32: return sizeof(float);
        case RenderDataType::UInt32: return sizeof(std::uint32_t);
    }
    return 0;
}

RenderBuffer::RenderBuffer(RenderOutputType output,
                           RenderDevice device,
                           int width,
                           int height,
                           std::shared_ptr<Storage> storage,
                           std::shared_ptr<void> frame_lease)
    : output_(output),
      device_(device),
      width_(width),
      height_(height),
      storage_(std::move(storage)),
      frame_lease_(std::move(frame_lease)) {}

RenderOutputType RenderBuffer::GetOutputType() const { return output_; }
RenderDataType RenderBuffer::GetDataType() const { return RenderOutputDataType(output_); }
RenderDevice RenderBuffer::GetDevice() const { return device_; }
int RenderBuffer::GetWidth() const { return width_; }
int RenderBuffer::GetHeight() const { return height_; }
std::size_t RenderBuffer::GetChannelCount() const { return RenderOutputChannelCount(output_); }
std::size_t RenderBuffer::GetByteSize() const {
    return static_cast<std::size_t>(width_) * height_ * GetChannelCount() *
           RenderDataTypeSize(GetDataType());
}
const void* RenderBuffer::GetData() const {
    return storage_ != nullptr && device_ == RenderDevice::Cpu ? storage_->bytes.data() : nullptr;
}
void* RenderBuffer::GetData() {
    return storage_ != nullptr && device_ == RenderDevice::Cpu ? storage_->bytes.data() : nullptr;
}

std::vector<std::size_t> RenderBuffer::GetShape() const {
    if (GetChannelCount() == 1) {
        return {static_cast<std::size_t>(height_), static_cast<std::size_t>(width_)};
    }
    return {static_cast<std::size_t>(height_),
            static_cast<std::size_t>(width_),
            GetChannelCount()};
}

bool RenderBuffer::CopyToHost(void* destination, std::size_t destination_size) const {
    if (storage_ == nullptr || destination == nullptr || destination_size != GetByteSize()) {
        return false;
    }
    if (device_ == RenderDevice::Cpu) {
        if (storage_->bytes.size() != destination_size) {
            return false;
        }
        std::memcpy(destination, storage_->bytes.data(), destination_size);
        return true;
    }
    return storage_->copy_to_host != nullptr &&
           storage_->copy_to_host(destination, destination_size);
}

std::shared_ptr<RenderBuffer> RenderFrame::Get(RenderOutputType output) const {
    const auto iter = buffers_.find(output);
    return iter == buffers_.end() ? nullptr : iter->second;
}

bool RenderFrame::Contains(RenderOutputType output) const { return buffers_.contains(output); }

std::vector<RenderOutputType> RenderFrame::GetOutputs() const {
    std::vector<RenderOutputType> outputs;
    outputs.reserve(buffers_.size());
    for (const auto& [output, buffer] : buffers_) {
        outputs.push_back(output);
    }
    return outputs;
}

std::uint64_t RenderFrame::GetFrameIndex() const { return frame_index_; }

const std::map<std::uint32_t, std::string>& RenderFrame::GetInstancePaths() const {
    return instance_paths_;
}

const std::map<std::uint32_t, std::string>& RenderFrame::GetSemanticLabels() const {
    return semantic_labels_;
}

RenderProduct::RenderProduct(RenderProductDesc desc) : desc_(std::move(desc)) {
    if (desc_.width <= 0 || desc_.height <= 0) {
        throw std::invalid_argument("render product dimensions must be positive");
    }
    if (desc_.frame_slots == 0) {
        throw std::invalid_argument("render product frame slot count must be positive");
    }
    if (desc_.outputs.empty()) {
        throw std::invalid_argument("render product must request at least one output");
    }

    std::unordered_set<std::uint32_t> seen;
    std::vector<RenderOutputType> unique_outputs;
    unique_outputs.reserve(desc_.outputs.size());
    for (const RenderOutputType output : desc_.outputs) {
        if (seen.insert(static_cast<std::uint32_t>(output)).second) {
            unique_outputs.push_back(output);
        }
    }
    desc_.outputs = std::move(unique_outputs);
    slots_.resize(desc_.frame_slots);
}

RenderProduct::~RenderProduct() {
    if (viewport_.IsValid() && RenderServer::HasInstance()) {
        RenderServer::GetInstance()->Free(viewport_);
    }
}

const RenderProductDesc& RenderProduct::GetDesc() const { return desc_; }
RenderDevice RenderProduct::GetDevice() const { return device_; }

void RenderProduct::EnsureDevice() {
    if (device_initialized_) {
        return;
    }
    if (!RenderServer::HasInstance()) {
        throw std::runtime_error("Gobot RenderServer is not initialized");
    }
    const SceneRendererCapabilities capabilities =
            RenderServer::GetInstance()->GetSceneRendererCapabilities();
    if (desc_.device == RenderDevice::Cuda && !capabilities.cuda_render_products) {
        throw std::runtime_error(
                "CUDA render products are unavailable; explicit device='cuda' does not fall back to CPU");
    }
    device_ = desc_.device == RenderDevice::Cuda ||
                      (desc_.device == RenderDevice::Auto && capabilities.cuda_render_products)
              ? RenderDevice::Cuda
              : RenderDevice::Cpu;
    device_initialized_ = true;
    if (device_ == RenderDevice::Cpu) {
        EnsureCpuViewport();
    }
}

void RenderProduct::EnsureCpuViewport() {
    if (viewport_.IsValid()) {
        return;
    }
    if (!RenderServer::HasInstance()) {
        throw std::runtime_error("Gobot RenderServer is not initialized");
    }
    RenderServer* render_server = RenderServer::GetInstance();
    viewport_ = render_server->ViewportCreate();
    std::uint32_t output_mask = 0;
    for (const RenderOutputType output : desc_.outputs) {
        output_mask |= RenderOutputBit(output);
    }
    render_server->ViewportSetOutputMask(viewport_, output_mask);
    render_server->ViewportSetSize(viewport_, desc_.width, desc_.height);
}

std::shared_ptr<RenderFrame> RenderProduct::Capture(const RenderSceneSnapshot& scene,
                                                    const RenderViewSnapshot& view) {
    EnsureDevice();
    auto available = std::find_if(slots_.begin(), slots_.end(), [](const FrameSlot& slot) {
        return slot.lease.expired();
    });
    if (available == slots_.end()) {
        throw std::runtime_error(
                "render product frame pool exhausted: all frame slots are still referenced");
    }

    std::uint32_t output_mask = 0;
    for (const RenderOutputType output : desc_.outputs) {
        output_mask |= RenderOutputBit(output);
    }

    RendererRenderProductFrame cuda_frame;
    if (device_ == RenderDevice::Cuda) {
        // The slot lease is expired, so its previous backend frame can be
        // released before allocating the replacement.
        available->storage.clear();
        std::string error;
        if (!RenderServer::GetInstance()->CaptureCudaRenderProduct(
                    scene,
                    view,
                    desc_.width,
                    desc_.height,
                    output_mask,
                    static_cast<std::uint32_t>(desc_.mode),
                    &cuda_frame,
                    &error)) {
            if (desc_.device == RenderDevice::Cuda) {
                throw std::runtime_error(
                        "CUDA render-product capture failed; explicit device='cuda' does not fall back: " +
                        error);
            }
            device_ = RenderDevice::Cpu;
            EnsureCpuViewport();
        }
    }

    auto lease = std::make_shared<std::uint64_t>(++frame_index_);
    available->lease = lease;
    auto frame = std::make_shared<RenderFrame>();
    frame->frame_lease_ = lease;
    frame->frame_index_ = frame_index_;
    frame->instance_paths_ = scene.instance_paths;
    frame->semantic_labels_ = scene.semantic_labels;

    RenderViewSnapshot capture_view = view;
    if (desc_.mode == RenderProductMode::Minimal) {
        capture_view.mode = RenderViewMode::Minimal;
    }

    RenderServer* render_server = RenderServer::GetInstance();
    if (device_ == RenderDevice::Cpu) {
        render_server->RenderSnapshotsToViewport(viewport_, scene, capture_view);
    }
    for (const RenderOutputType output : desc_.outputs) {
        std::shared_ptr<RenderBuffer::Storage>& storage = available->storage[output];
        if (storage == nullptr) {
            storage = std::make_shared<RenderBuffer::Storage>();
        }
        const std::size_t byte_size = static_cast<std::size_t>(desc_.width) * desc_.height *
                                      RenderOutputChannelCount(output) *
                                      RenderDataTypeSize(RenderOutputDataType(output));
        if (device_ == RenderDevice::Cpu) {
            storage->device_pointer = nullptr;
            storage->allocation_size = 0;
            storage->pixel_stride_bytes = 0;
            storage->device_id = 0;
            storage->backend_owner.reset();
            storage->copy_to_host = {};
            storage->bytes.resize(byte_size);
            if (!render_server->ReadViewportOutput(
                        viewport_, output, storage->bytes.data(), storage->bytes.size(), true)) {
                throw std::runtime_error(std::string("failed to read render output '") +
                                         RenderOutputTypeName(output) + "'");
            }
        } else {
            const std::uint32_t index = static_cast<std::uint32_t>(output);
            const RendererRenderProductBuffer& backend_buffer = cuda_frame.buffers[index];
            if (backend_buffer.device_pointer == nullptr || backend_buffer.allocation_size == 0) {
                throw std::runtime_error(std::string("CUDA renderer did not return output '") +
                                         RenderOutputTypeName(output) + "'");
            }
            storage->bytes.clear();
            storage->device_pointer = backend_buffer.device_pointer;
            storage->allocation_size = backend_buffer.allocation_size;
            storage->pixel_stride_bytes = backend_buffer.pixel_stride_bytes;
            storage->device_id = cuda_frame.device_id;
            storage->backend_owner = cuda_frame.owner;
            storage->copy_to_host = [copy = cuda_frame.copy_to_host, index](void* destination,
                                                                            std::size_t size) {
                return copy != nullptr && copy(index, destination, size);
            };
        }
        frame->buffers_.emplace(output,
                                std::shared_ptr<RenderBuffer>(new RenderBuffer(
                                        output,
                                        device_,
                                        desc_.width,
                                        desc_.height,
                                        storage,
                                        lease)));
    }
    return frame;
}

CameraSensor::CameraSensor(Camera3D* camera, Node* root, RenderProductDesc desc)
    : camera_id_(camera != nullptr ? camera->GetInstanceId() : ObjectID()),
      root_id_(root != nullptr ? root->GetInstanceId() : ObjectID()),
      render_product_(std::make_shared<RenderProduct>(std::move(desc))) {
    if (camera == nullptr || root == nullptr) {
        throw std::invalid_argument("camera sensor requires a camera and scene root");
    }
}

std::shared_ptr<RenderFrame> CameraSensor::Capture() {
    auto* camera = Object::PointerCastTo<Camera3D>(ObjectDB::GetInstance(camera_id_));
    auto* root = Object::PointerCastTo<Node>(ObjectDB::GetInstance(root_id_));
    if (camera == nullptr || root == nullptr) {
        throw std::runtime_error("camera sensor camera or scene root is no longer valid");
    }

    RenderSceneSnapshot scene = CaptureRenderSceneSnapshot(root);
    RenderViewSnapshot view = CaptureRenderViewSnapshot(*camera);
    const RenderProductDesc& desc = render_product_->GetDesc();
    view.camera.projection = Matrix4f::Perspective(camera->GetFovy(),
                                                   static_cast<RealType>(desc.width) / desc.height,
                                                   camera->GetNear(),
                                                   camera->GetFar());
    view.camera.view_projection = view.camera.projection * view.camera.view;
    return render_product_->Capture(scene, view);
}

std::shared_ptr<RenderProduct> CameraSensor::GetRenderProduct() const { return render_product_; }

} // namespace gobot
