#include "gobot/rendering/render_scene_preparation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>

namespace gobot {
namespace {

RenderFrustumPlane MakePlane(const Vector4& coefficients, bool* valid) {
    const Vector3 normal = coefficients.template head<3>();
    const RealType length = normal.norm();
    if (!coefficients.allFinite() || length <= CMP_EPSILON) {
        *valid = false;
        return {};
    }
    return {normal / length, coefficients.w() / length};
}

RealType CameraDepth(const VisualMeshRenderItem& item, const Matrix4& view) {
    Vector3 center = item.world_bounds.IsValid()
                             ? item.world_bounds.GetCenter()
                             : item.model.template block<3, 1>(0, 3);
    const Vector4 view_center = view * Vector4{center.x(), center.y(), center.z(), 1.0};
    return std::max<RealType>(0.0, -view_center.z());
}

auto StableItemKey(const PreparedRenderItem& prepared) {
    const VisualMeshRenderItem& item = *prepared.item;
    return std::tuple{item.material.material_id.operator std::uint64_t(),
                      item.mesh_id.operator std::uint64_t(),
                      item.surface_index,
                      item.instance_id};
}

} // namespace

RenderFrustum RenderFrustum::FromViewProjection(const Matrix4& matrix) {
    RenderFrustum result;
    if (!matrix.allFinite()) {
        return result;
    }

    const Vector4 row0 = matrix.row(0).transpose();
    const Vector4 row1 = matrix.row(1).transpose();
    const Vector4 row2 = matrix.row(2).transpose();
    const Vector4 row3 = matrix.row(3).transpose();
    bool valid = true;
    result.planes_ = {
            MakePlane(row3 + row0, &valid),
            MakePlane(row3 - row0, &valid),
            MakePlane(row3 + row1, &valid),
            MakePlane(row3 - row1, &valid),
            MakePlane(row3 + row2, &valid),
            MakePlane(row3 - row2, &valid)};
    result.valid_ = valid;
    return result;
}

bool RenderFrustum::Intersects(const AABB& bounds) const {
    if (!valid_ || !bounds.IsValid()) {
        return true;
    }
    const Vector3 center = bounds.GetCenter();
    const Vector3 extents = bounds.GetExtents();
    for (const RenderFrustumPlane& plane : planes_) {
        const RealType radius = plane.normal.cwiseAbs().dot(extents);
        if (plane.normal.dot(center) + plane.distance + radius < 0.0) {
            return false;
        }
    }
    return true;
}

bool RenderFrustum::IsValid() const { return valid_; }

RenderDrawLists BuildRenderDrawLists(const RenderSceneSnapshot& scene,
                                     const RenderViewSnapshot& view,
                                     bool frustum_culling) {
    RenderDrawLists lists;
    const RenderFrustum frustum = RenderFrustum::FromViewProjection(view.camera.view_projection);
    for (const VisualMeshRenderItem& item : scene.visual_meshes) {
        PreparedRenderItem prepared{&item, CameraDepth(item, view.camera.view)};
        if (item.material.alpha_mode != AlphaMode::Blend) {
            lists.shadow_casters.emplace_back(prepared);
        }
        if (frustum_culling && !frustum.Intersects(item.world_bounds)) {
            ++lists.culled_count;
            continue;
        }

        ++lists.visible_count;
        switch (item.material.alpha_mode) {
            case AlphaMode::Opaque:
                lists.opaque.emplace_back(prepared);
                break;
            case AlphaMode::Mask:
                lists.alpha_masked.emplace_back(prepared);
                break;
            case AlphaMode::Blend:
                lists.transparent.emplace_back(prepared);
                break;
        }
    }

    const auto front_to_back = [](const PreparedRenderItem& left, const PreparedRenderItem& right) {
        if (left.camera_depth != right.camera_depth) {
            return left.camera_depth < right.camera_depth;
        }
        return StableItemKey(left) < StableItemKey(right);
    };
    const auto back_to_front = [](const PreparedRenderItem& left, const PreparedRenderItem& right) {
        if (left.camera_depth != right.camera_depth) {
            return left.camera_depth > right.camera_depth;
        }
        return StableItemKey(left) < StableItemKey(right);
    };
    std::stable_sort(lists.opaque.begin(), lists.opaque.end(), front_to_back);
    std::stable_sort(lists.alpha_masked.begin(), lists.alpha_masked.end(), front_to_back);
    std::stable_sort(lists.transparent.begin(), lists.transparent.end(), back_to_front);
    std::stable_sort(lists.shadow_casters.begin(), lists.shadow_casters.end(), front_to_back);
    return lists;
}

} // namespace gobot
