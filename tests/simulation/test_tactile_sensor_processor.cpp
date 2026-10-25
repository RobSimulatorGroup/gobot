#include <gtest/gtest.h>
#include "gobot/simulation/tactile_sensor_processor.hpp"
#include "gobot/core/math/mesh_validation.hpp"
#include "gobot/scene/tactile_sensor_3d.hpp"
#include "gobot/scene/resources/packed_scene.hpp"
#include <numeric>
using namespace gobot;
namespace {
struct Fixture {
    std::vector<Vector3> vertices{{-.01, -.01, 0}, {.01, -.01, 0}, {-.01, .01, 0}, {-.01, -.01, -.004}};
    std::vector<std::uint32_t> tets{0, 2, 1, 3};
    TactileSensorParameters p;
    Fixture() {
        p.image_width = p.image_height = 32;
        p.pixel_size = .001;
        p.far_plane = .1;
        p.camera_transform.translation() = Vector3{0, 0, -.03};
        p.coat_vertex_indices = {0, 1, 2};
        p.stick_vertex_indices = {3};
        p.marker_positions = {Vector2{12.666667, 19.333333}};
        p.marker_tetrahedra = {0};
        p.marker_barycentric = {Vector4{1. / 3, 1. / 3, 1. / 3, 0}};
    }
    TactileSensorProcessor Make(TactileOpticalData c = {}) {
        return {p, vertices, tets, ResolveTetrahedralSurface(tets, {}), std::move(c)};
    }
};
} // namespace
TEST(TactileProcessor, RestAndRigidMotionDoNotProduceIndentationOrMarkerFlow) {
    Fixture f;
    auto processor = f.Make();
    auto pose = Affine3::Identity();
    pose.linear() = Eigen::AngleAxis<RealType>(.7, Vector3::UnitZ()).toRotationMatrix();
    pose.translation() = Vector3{1, 2, 3};
    auto vertices = f.vertices;
    for (auto &v : vertices)
        v = pose * v;
    auto frame = processor.Process(vertices, std::vector<Vector3>(4, Vector3::Zero()), pose, 4, .1);
    for (auto v : frame->indentation)
        EXPECT_NEAR(v, 0, 1e-6);
    EXPECT_LT(frame->marker_motion.front().norm(), 1e-3);
    EXPECT_TRUE(frame->total_force.isZero());
    EXPECT_EQ(frame->tick, 4);
}
TEST(TactileProcessor, PressureAndShearConserveCoatForcesAndIgnoreBackingForce) {
    Fixture f;
    auto processor = f.Make();
    auto vertices = f.vertices;
    for (int i = 0; i < 3; ++i)
        vertices[i].z() -= .0002;
    std::vector<Vector3> forces(4, Vector3::Zero());
    forces[0] = {.3, .2, -2};
    forces[3] = {50, 40, 100};
    auto frame = processor.Process(vertices, forces, Affine3::Identity(), 1, .01);
    Vector3 integral = Vector3::Zero();
    const auto &areas = processor.GetPixelAreas();
    for (std::size_t i = 0; i < areas.size(); ++i) {
        integral +=
            Vector3{frame->shear_stress[i * 2], frame->shear_stress[i * 2 + 1], -frame->normal_pressure[i]} *
            areas[i];
    }
    EXPECT_TRUE(integral.isApprox(forces[0], 1e-5));
    EXPECT_TRUE(frame->total_force.isApprox(forces[0], 1e-6));
    EXPECT_NEAR(*std::max_element(frame->indentation.begin(), frame->indentation.end()), .0002, 1e-7);
    auto retained = frame;
    processor.Process(f.vertices, std::vector<Vector3>(4, Vector3::Zero()), Affine3::Identity(), 2, .02);
    EXPECT_EQ(retained->tick, 1);
    EXPECT_NEAR(retained->total_force.z(), -2, 1e-6);
}
TEST(TactileProcessor, TaximUsesCalibrationPolynomialAndBackground) {
    Fixture f;
    f.p.rgb_model = "taxim_v1";
    TactileOpticalData c;
    c.width = c.height = 32;
    c.bins = 2;
    c.coefficients.assign(3 * 2 * 2 * 6, 0);
    c.background.assign(32 * 32 * 3, .2);
    for (std::size_t i = 5; i < c.coefficients.size(); i += 6)
        c.coefficients[i] = .1;
    auto processor = f.Make(c);
    auto frame =
        processor.Process(f.vertices, std::vector<Vector3>(4, Vector3::Zero()), Affine3::Identity(), 0, 0);
    for (auto value : frame->rgb)
        EXPECT_NEAR(int(value), 77, 1);
    c.coefficients[0] = std::numeric_limits<RealType>::quiet_NaN();
    EXPECT_THROW(f.Make(c), std::invalid_argument);
}
TEST(TactileProcessor, InvalidStateAndOpticsAreRejected) {
    Fixture f;
    auto processor = f.Make();
    EXPECT_THROW(processor.Process(f.vertices, {}, Affine3::Identity(), 0, 0), std::invalid_argument);
    f.p.rgb_model = "unknown";
    EXPECT_THROW(f.Make(), std::invalid_argument);
}
TEST(TactileProcessor, ConfigSurvivesPackedSceneInstantiation) {
    auto *sensor = Object::New<TactileSensor3D>();
    sensor->SetName("Touch");
    auto config = MakeRef<TactileSensorConfig>();
    config->SetCameraIntrinsics(Vector4{320, 320, 160, 120});
    config->SetAttachmentStrength(10000);
    auto optics = MakeRef<TactileOpticalCalibration>();
    optics->SetBins(12);
    config->SetOpticalCalibration(optics);
    sensor->SetConfig(config);
    auto packed = MakeRef<PackedScene>();
    ASSERT_TRUE(packed->Pack(sensor));
    auto *clone = Object::PointerCastTo<TactileSensor3D>(packed->Instantiate());
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->GetConfig()->GetCameraIntrinsics(), config->GetCameraIntrinsics());
    ASSERT_TRUE(clone->GetConfig()->GetOpticalCalibration().IsValid());
    EXPECT_EQ(clone->GetConfig()->GetOpticalCalibration()->GetBins(), 12);
    EXPECT_FALSE(clone->GetRuntimeFrame());
    Object::Delete(clone);
    Object::Delete(sensor);
}

TEST(TactileProcessor, OpticalNormalsVaryContinuouslyWithinCoarseFemFaces) {
    Fixture f;
    f.vertices = {{-.01, -.01, 0}, {.01, -.01, 0}, {.01, .01, 0}, {-.01, .01, 0}, {0, 0, 0}, {0, 0, -.004}};
    f.tets.clear();
    for (std::uint32_t i = 0; i < 4; ++i)
        f.tets.insert(f.tets.end(), {i, 4, (i + 1) % 4, 5});
    f.p.coat_vertex_indices = {0, 1, 2, 3, 4};
    f.p.stick_vertex_indices = {5};
    f.p.marker_positions.clear();
    f.p.marker_tetrahedra.clear();
    f.p.marker_barycentric.clear();
    auto processor = f.Make();
    auto deformed = f.vertices;
    deformed[4].z() = -.003;
    auto frame =
        processor.Process(deformed, std::vector<Vector3>(6, Vector3::Zero()), Affine3::Identity(), 1, .01);
    int low = 255, high = 0;
    for (int x = 17; x <= 23; ++x) {
        const auto value = frame->rgb[(16 * 32 + x) * 3];
        low = std::min(low, int(value));
        high = std::max(high, int(value));
    }
    // A planar-face normal would give a constant color on this ray. A smooth
    // optical surface must interpolate between its vertex normals instead.
    EXPECT_GT(high - low, 3);
    EXPECT_GT(*std::max_element(frame->indentation.begin(), frame->indentation.end()), .002);
    EXPECT_TRUE(frame->total_force.isZero());
}

TEST(TactileProcessor, ColliderDepthPreservesAHoleWhileFemForceRemainsIndependent) {
    Fixture f;
    f.vertices = {{-.01, -.01, 0}, {.01, -.01, 0}, {.01, .01, 0}, {-.01, .01, 0}, {0, 0, 0}, {0, 0, -.004}};
    f.tets.clear();
    for (std::uint32_t i = 0; i < 4; ++i)
        f.tets.insert(f.tets.end(), {i, 4, (i + 1) % 4, 5});
    f.p.coat_vertex_indices = {0, 1, 2, 3, 4};
    f.p.stick_vertex_indices = {5};
    f.p.marker_positions.clear();
    f.p.marker_tetrahedra.clear();
    f.p.marker_barycentric.clear();
    f.p.image_width = f.p.image_height = 128;
    f.p.pixel_size = .0002;
    auto processor = f.Make();
    std::vector<Vector3> ring;
    std::vector<std::uint32_t> indices;
    for (int i = 0; i < 64; ++i) {
        const double a = 2 * 3.141592653589793 * i / 64;
        ring.push_back(Vector3{RealType(.006 * std::cos(a)), RealType(.006 * std::sin(a)), -.001f});
        ring.push_back(Vector3{RealType(.004 * std::cos(a)), RealType(.004 * std::sin(a)), -.001f});
        const auto next = std::uint32_t((i + 1) % 64) * 2, current = std::uint32_t(i) * 2;
        indices.insert(indices.end(), {current, next, current + 1, next, next + 1, current + 1});
    }
    auto frame = processor.Process(f.vertices, std::vector<Vector3>(6, Vector3::Zero()), Affine3::Identity(),
                                   1, .01, ring, indices);
    const auto center = 64 * 128 + 64, rim = 64 * 128 + 89;
    EXPECT_NEAR(frame->contact_depth[center], f.p.far_plane, 1e-6);
    EXPECT_NEAR(frame->contact_depth[rim], .029, 1e-6);
    EXPECT_LT(frame->optical_indentation[center], frame->optical_indentation[rim] * .25);
    EXPECT_GT(frame->optical_indentation[rim], .0008);
    EXPECT_TRUE(frame->total_force.isZero());
    for (auto v : frame->indentation)
        EXPECT_NEAR(v, 0, 1e-7);
    for (auto &v : ring)
        v.z() = .003;
    auto released = processor.Process(f.vertices, std::vector<Vector3>(6, Vector3::Zero()),
                                      Affine3::Identity(), 2, .02, ring, indices);
    EXPECT_EQ(*std::max_element(released->optical_indentation.begin(), released->optical_indentation.end()),
              0);
}
