#include "luisa_renderer_internal.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace gobot::luisa_renderer {

GeometryResource* LuisaRenderer::EnsureGeometry(const gobot::VisualMeshRenderItem& item,
                                                std::string* error) {
        const gobot::MeshSurfaceData* surface = item.GetSurface();
        if (surface == nullptr) {
            *error = "Render snapshot contains an invalid mesh surface.";
            return nullptr;
        }
        const GeometryKey key{
                item.mesh_id.operator std::uint64_t(), item.mesh_topology_revision, item.surface_index};
        auto [found, inserted] = geometry_cache_.try_emplace(key);
        if (inserted) {
            found->second = std::make_unique<GeometryResource>();
        }
        auto* resource = found->second.get();
        if (!inserted && resource->geometry_revision == item.mesh_revision) {
            return resource;
        }
        const auto upload_start = std::chrono::steady_clock::now();

        auto& vertices = resource->staging_vertices;
        vertices.clear();
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
        if (vertices.empty() || surface->indices.empty() || surface->indices.size() % 3 != 0) {
            *error = "Render snapshot contains empty geometry.";
            geometry_cache_.erase(found);
            return nullptr;
        }
        if (inserted) {
            auto& triangles = resource->staging_triangles;
            triangles.reserve(surface->indices.size() / 3);
            for (std::size_t i = 0; i < surface->indices.size(); i += 3) {
                triangles.push_back({surface->indices[i], surface->indices[i + 1], surface->indices[i + 2]});
            }
            resource->vertices = device_->create_buffer<GpuVertex>(vertices.size());
            resource->triangles = device_->create_buffer<Triangle>(triangles.size());
            resource->mesh = device_->create_mesh(resource->vertices, resource->triangles, {.allow_update = true});
            *stream_ << resource->triangles.copy_from(luisa::span{triangles});
            resource_stats_.uploaded_bytes += triangles.size() * sizeof(Triangle);
            ++resource_stats_.index_uploads;
        }
        *stream_ << resource->vertices.copy_from(luisa::span{vertices})
                 << resource->mesh.build();
        resource->geometry_revision = item.mesh_revision;
        resource_stats_.uploaded_bytes += vertices.size() * sizeof(GpuVertex);
        ++resource_stats_.geometry_uploads;
        resource_stats_.upload_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - upload_start).count();
        return resource;
    }

std::uint32_t LuisaRenderer::BindTexture(const gobot::RenderTextureSnapshot& texture) {
        if (!texture.IsValid() || next_texture_slot_ >= 65535u) {
            return kInvalidTexture;
        }
        const TextureKey key{
                texture.image.image_id.operator std::uint64_t(),
                texture.image.revision};
        TextureResource* resource = nullptr;
        if (const auto found = texture_cache_.find(key); found != texture_cache_.end()) {
            resource = found->second.get();
        } else {
            const auto upload_start = std::chrono::steady_clock::now();
            auto created = std::make_unique<TextureResource>();
            auto& pixels = created->staging_pixels;
            pixels = ConvertImage(*texture.image.storage);
            if (pixels.empty()) {
                return kInvalidTexture;
            }
            created->image = device_->create_image<float>(
                    PixelStorage::FLOAT4,
                    make_uint2(static_cast<uint>(texture.image.storage->width),
                               static_cast<uint>(texture.image.storage->height)));
            *stream_ << created->image.copy_from(luisa::span{pixels});
            created->resident_bytes = pixels.size() * sizeof(float4);
            resource_stats_.uploaded_bytes += created->resident_bytes;
            ++resource_stats_.image_uploads;
            resource_stats_.upload_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - upload_start).count();
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
        accel_ = device_->create_accel({.allow_update = true});
        geometry_heap_ = device_->create_bindless_array(
                std::max<std::size_t>(2, snapshot.visual_meshes.size() * 2));
        if (snapshot.visual_meshes.empty()) {
            // OptiX requires a nonempty acceleration structure even for a background-only frame.
            // The internal triangle is masked out for every ray and is not a scene asset.
            if (!empty_geometry_) {
                empty_geometry_ = std::make_unique<GeometryResource>();
                auto& placeholder = *empty_geometry_;
                placeholder.staging_vertices.resize(3);
                placeholder.staging_vertices[0].position = make_float3(0.0f, 0.0f, 0.0f);
                placeholder.staging_vertices[1].position = make_float3(1.0f, 0.0f, 0.0f);
                placeholder.staging_vertices[2].position = make_float3(0.0f, 1.0f, 0.0f);
                placeholder.staging_triangles.push_back({0, 1, 2});
                placeholder.vertices = device_->create_buffer<GpuVertex>(3);
                placeholder.triangles = device_->create_buffer<Triangle>(1);
                placeholder.mesh = device_->create_mesh(placeholder.vertices, placeholder.triangles);
                *stream_ << placeholder.vertices.copy_from(luisa::span{placeholder.staging_vertices})
                         << placeholder.triangles.copy_from(luisa::span{placeholder.staging_triangles})
                         << placeholder.mesh.build();
            }
            geometry_heap_.emplace_on_update(0, empty_geometry_->vertices);
            geometry_heap_.emplace_on_update(1, empty_geometry_->triangles);
            accel_.emplace_back(empty_geometry_->mesh, make_float4x4(1.0f), 0u, true, 0u);
        }
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
        const bool topology_changed = last_topology_ != snapshot.fingerprints.topology;
        const bool geometry_changed = last_geometry_ != snapshot.fingerprints.geometry;
        const bool materials_changed = last_materials_ != snapshot.fingerprints.materials ||
                                       last_lighting_ != snapshot.fingerprints.lighting;
        if (topology_changed || geometry_changed || materials_changed) {
            *stream_ << synchronize();
        }
        if (topology_changed && !RebuildTopology(snapshot, error)) {
            *stream_ << synchronize();
            active_geometry_.clear();
            accel_ = {};
            geometry_heap_ = {};
            geometry_cache_.clear();
            last_topology_ = 0;
            return false;
        }
        if (!topology_changed && geometry_changed) {
            for (const auto& item : snapshot.visual_meshes) {
                if (EnsureGeometry(item, error) == nullptr) {
                    *stream_ << synchronize();
                    last_topology_ = 0;
                    return false;
                }
            }
        }
        if (!topology_changed && (geometry_changed || last_transforms_ != snapshot.fingerprints.transforms)) {
            UpdateTransforms(snapshot);
        }
        if (topology_changed || materials_changed) {
            UpdateMaterialsAndLighting(snapshot);
        }
        if (topology_changed || geometry_changed || materials_changed) {
            PruneSceneCaches(snapshot);
        }
        last_topology_ = snapshot.fingerprints.topology;
        last_geometry_ = snapshot.fingerprints.geometry;
        last_transforms_ = snapshot.fingerprints.transforms;
        last_materials_ = snapshot.fingerprints.materials;
        last_lighting_ = snapshot.fingerprints.lighting;
        if (snapshot.visual_meshes.empty() && !allow_empty) {
            *error = "Scene has no renderable mesh; using raster fallback.";
            return false;
        }
        return true;
}

void LuisaRenderer::PruneSceneCaches(const RenderSceneSnapshot& snapshot) {
    // Every update above has completed. Old heaps/accels no longer refer to these resources.
    std::unordered_set<GeometryKey, GeometryKeyHash> meshes;
    std::unordered_set<TextureKey, TextureKeyHash> images;
    const auto retain_texture = [&](const RenderTextureSnapshot& texture) {
        if (texture.IsValid()) {
            images.insert({texture.image.image_id.operator std::uint64_t(), texture.image.revision});
        }
    };
    for (const auto& item : snapshot.visual_meshes) {
        meshes.insert({item.mesh_id.operator std::uint64_t(), item.mesh_topology_revision, item.surface_index});
        retain_texture(item.material.albedo_texture);
        retain_texture(item.material.metallic_roughness_texture);
        retain_texture(item.material.normal_texture);
        retain_texture(item.material.occlusion_texture);
        retain_texture(item.material.emissive_texture);
    }
    retain_texture(snapshot.environment.environment_texture);
    std::erase_if(geometry_cache_, [&](const auto& entry) { return !meshes.contains(entry.first); });
    std::erase_if(texture_cache_, [&](const auto& entry) { return !images.contains(entry.first); });
    for (auto& [key, resource] : geometry_cache_) {
        resource->staging_vertices.clear();
        resource->staging_triangles.clear();
    }
    for (auto& [key, resource] : texture_cache_) {
        resource->staging_pixels.clear();
    }
}

RenderResourceStats LuisaRenderer::GetResourceStats() const {
    auto stats = resource_stats_;
    stats.mesh_entries = geometry_cache_.size();
    stats.texture_entries = texture_cache_.size();
    for (const auto& [key, resource] : geometry_cache_) {
        stats.resident_bytes += resource->vertices.size_bytes() + resource->triangles.size_bytes();
    }
    for (const auto& [key, resource] : texture_cache_) {
        stats.resident_bytes += resource->resident_bytes;
    }
    return stats;
}

} // namespace gobot::luisa_renderer
