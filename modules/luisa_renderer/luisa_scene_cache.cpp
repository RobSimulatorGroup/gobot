#include "luisa_renderer_internal.hpp"

#include <algorithm>
#include <cmath>

namespace gobot::luisa_renderer {

GeometryResource* LuisaRenderer::EnsureGeometry(const gobot::VisualMeshRenderItem& item,
                                                std::string* error) {
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

std::uint32_t LuisaRenderer::BindTexture(const gobot::RenderTextureSnapshot& texture) {
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

GpuMaterial LuisaRenderer::MakeMaterial(const gobot::RenderMaterialSnapshot& material) {
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

bool LuisaRenderer::RebuildTopology(const gobot::RenderSceneSnapshot& snapshot,
                                    std::string* error) {
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
            const auto visibility = static_cast<std::uint8_t>(
                    (snapshot.visual_meshes[i].visible_in_rgb ? kRgbVisibility : 0u) |
                    (snapshot.visual_meshes[i].cast_shadow ? kShadowVisibility : 0u) |
                    kAovVisibility);
            accel_.emplace_back(geometry->mesh,
                                ToLuisaMatrix(snapshot.visual_meshes[i].model),
                                visibility,
                                snapshot.visual_meshes[i].material.alpha_mode == gobot::AlphaMode::Opaque,
                                static_cast<uint>(i));
        }
        *stream_ << geometry_heap_.update()
                 << accel_.build(AccelBuildRequest::FORCE_BUILD)
                 << synchronize();
        return true;
    }

void LuisaRenderer::UpdateTransforms(const gobot::RenderSceneSnapshot& snapshot) {
        for (std::size_t i = 0; i < snapshot.visual_meshes.size(); ++i) {
            accel_.set_transform_on_update(i, ToLuisaMatrix(snapshot.visual_meshes[i].model));
        }
        *stream_ << accel_.build(AccelBuildRequest::PREFER_UPDATE) << synchronize();
    }

void LuisaRenderer::UpdateMaterialsAndLighting(const gobot::RenderSceneSnapshot& snapshot) {
        texture_heap_ = device_->create_bindless_array(65536);
        next_texture_slot_ = 0;
        std::vector<GpuMaterial> host_materials;
        std::vector<uint> host_instance_ids;
        std::vector<uint> host_semantic_ids;
        std::vector<uint> host_rgb_modes;
        host_materials.reserve(std::max<std::size_t>(1, snapshot.visual_meshes.size()));
        host_instance_ids.reserve(std::max<std::size_t>(1, snapshot.visual_meshes.size()));
        host_semantic_ids.reserve(std::max<std::size_t>(1, snapshot.visual_meshes.size()));
        host_rgb_modes.reserve(std::max<std::size_t>(1, snapshot.visual_meshes.size()));
        for (const gobot::VisualMeshRenderItem& item : snapshot.visual_meshes) {
            host_materials.emplace_back(MakeMaterial(item.material));
            host_instance_ids.emplace_back(item.instance_id);
            host_semantic_ids.emplace_back(item.semantic_id);
            host_rgb_modes.emplace_back(item.visible_in_rgb ? 1u : 2u);
        }
        if (host_materials.empty()) {
            host_materials.emplace_back(MakeMaterial({}));
            host_instance_ids.emplace_back(0u);
            host_semantic_ids.emplace_back(0u);
            host_rgb_modes.emplace_back(1u);
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
        instance_ids_ = device_->create_buffer<uint>(host_instance_ids.size());
        semantic_ids_ = device_->create_buffer<uint>(host_semantic_ids.size());
        rgb_modes_ = device_->create_buffer<uint>(host_rgb_modes.size());
        *stream_ << materials_.copy_from(luisa::span{host_materials})
                 << lights_.copy_from(luisa::span{host_lights})
                 << instance_ids_.copy_from(luisa::span{host_instance_ids})
                 << semantic_ids_.copy_from(luisa::span{host_semantic_ids});
        *stream_ << rgb_modes_.copy_from(luisa::span{host_rgb_modes});
        if (next_texture_slot_ != 0u) {
            *stream_ << texture_heap_.update();
        }
        *stream_ << synchronize();
    }

bool LuisaRenderer::SyncScene(const gobot::RenderSceneSnapshot& snapshot,
                              std::string* error,
                              bool allow_empty) {
        if (snapshot.visual_meshes.empty() && !allow_empty) {
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

} // namespace gobot::luisa_renderer
