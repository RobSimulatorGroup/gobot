// Host-staged experiment only; the normal Gobot solver/provider ABI is unchanged.
#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <uipc/constitution/affine_body_prismatic_joint.h>
#include <uipc/constitution/external_articulation_constraint.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/core/solver_diagnostics_feature.h>
#include <uipc/geometry/utils/affine_body/affine_body_from_rigid_body.h>
#include <uipc/geometry/utils/affine_body/transform.h>
#include <uipc/geometry/utils/label_surface.h>
#include <uipc/geometry/utils/label_triangle_orient.h>
#include <uipc/common/log.h>
#include <Eigen/Cholesky>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
using namespace uipc::constitution;
using Json = nlohmann::json;
thread_local std::string error;

Matrix4x4 Transform(const Json& data) {
    const auto values = data.at("matrix_row_major").get<std::array<double, 16>>();
    return Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(values.data());
}

template <class T>
std::vector<T> Vectors(const Json& data) {
    std::vector<T> values;
    for (const auto& row : data) {
        T value;
        for (int i = 0; i < value.size(); ++i) value[i] = row.at(i).get<typename T::Scalar>();
        values.push_back(value);
    }
    return values;
}

struct Probe {
    std::unique_ptr<Engine> engine;
    // Reverse member destruction also preserves SDK ownership if construction fails.
    std::unique_ptr<Scene> scene;
    std::unique_ptr<World> world;
    S<GeometrySlot> articulation;
    S<SimplicialComplexSlot> soft;
    std::vector<S<SimplicialComplexSlot>> moving;
    std::string result;
    std::size_t dofs{};

    explicit Probe(const Json& job) {
        uipc::logger::set_level(spdlog::level::warn);
        engine = std::make_unique<Engine>("cuda", job.at("workspace").get<std::string>());
        world = std::make_unique<World>(*engine);
        auto config = Scene::default_config();
        config["dt"] = job.at("dt").get<double>();
        config["gravity"] = Vector3{0., 0., -9.81};
        config["contact"]["enable"] = job.at("contact").get<bool>();
        config["contact"]["d_hat"] = .0008;
        config["newton"]["max_iter"] = 48;
        config["newton"]["velocity_tol"] = .1;
        config["newton"]["transrate_tol"] = 10.;
        config["newton"]["ccd_tol"] = 5.e-4;
        config["line_search"]["max_iter"] = 16;
        config["linear_system"]["tol_rate"] = job.at("linear_tolerance").get<double>();
        config["extras"]["strict_mode"]["enable"] = true;
        scene = std::make_unique<Scene>(config);
        scene->contact_tabular().default_model(1., 1.e9);
        auto rigid_contact = scene->contact_tabular().create("probe_rigid");
        scene->contact_tabular().insert(rigid_contact, rigid_contact, 0., 0., false);

        const auto& body = job.at("soft");
        auto mesh = tetmesh(Vectors<Vector3>(body.at("vertices")),
                            Vectors<Vector4i>(body.at("tetrahedra")));
        label_surface(mesh);
        label_triangle_orient(mesh);
        StableNeoHookean{}.apply_to(mesh, ElasticModuli::youngs_poisson(
                body.at("young_modulus"), body.at("poisson_ratio")), body.at("density"));
        soft = scene->objects().create("soft")->geometries().create(mesh).geometry;

        std::unordered_map<std::string, S<SimplicialComplexSlot>> links;
        for (const auto& link : job.at("links")) {
            auto rigid = trimesh(Vectors<Vector3>(link.at("vertices")),
                                  Vectors<Vector3i>(link.at("triangles")));
            label_surface(rigid);
            Matrix3x3 inertia = Vector3(link.at("inertia_diagonal")[0],
                    link.at("inertia_diagonal")[1], link.at("inertia_diagonal")[2]).asDiagonal();
            const auto mass = affine_body::from_rigid_body(link.at("mass"), Vector3::Zero(), inertia);
            AffineBodyConstitution{}.apply_to(rigid, 1.e8, mass, .001);
            view(rigid.transforms())[0] = Transform(link.at("transform"));
            view(*rigid.instances().find<IndexT>(builtin::is_fixed))[0] = link.at("fixed").get<bool>();
            view(*rigid.instances().find<IndexT>(builtin::external_kinetic))[0] = 1;
            auto previous = rigid.instances().create<Vector12>("ref_dof_prev");
            view(*previous)[0] = affine_body::transform_to_q(rigid.transforms().view()[0]);
            rigid_contact.apply_to(rigid);
            auto object = scene->objects().create(link.at("path").get<std::string>());
            auto slot = object->geometries().create(rigid).geometry;
            links.emplace(link.at("path").get<std::string>(), slot);
            if (!link.at("fixed").get<bool>()) moving.push_back(slot);
            scene->animator().insert(*object, [](Animation::UpdateInfo& info) {
                auto geo = info.geo_slots()[0]->geometry().as<SimplicialComplex>();
                view(*geo->instances().find<Vector12>("ref_dof_prev"))[0] =
                        affine_body::transform_to_q(geo->transforms().view()[0]);
            });
        }
        std::vector<S<const GeometrySlot>> joints;
        for (const auto& joint : job.at("joints")) {
            if (joint.at("joint_type").get<int>() != 3) {
                throw std::runtime_error("probe accepts scalar prismatic joints only");
            }
            auto transform = Transform(joint.at("transform"));
            Vector3 axis(joint.at("axis")[0], joint.at("axis")[1], joint.at("axis")[2]);
            axis = transform.block<3, 3>(0, 0) * axis;
            const Vector3 origin = transform.block<3, 1>(0, 3);
            const std::array<Vector3, 1> start{origin - .05 * axis}, end{origin + .05 * axis};
            std::array<S<SimplicialComplexSlot>, 1> parent{links.at(joint.at("parent_link_path"))};
            std::array<S<SimplicialComplexSlot>, 1> child{links.at(joint.at("child_link_path"))};
            std::array<IndexT, 1> indices{0};
            std::array<Float, 1> strength{100.};
            auto geometry = AffineBodyPrismaticJoint{}.create_geometry(start, end, parent, indices,
                                                                       child, indices, strength);
            auto object = scene->objects().create(joint.at("path").get<std::string>());
            joints.push_back(object->geometries().create(geometry).geometry);
        }
        dofs = joints.size();
        if (dofs != 2) throw std::runtime_error("probe requires exactly two serial joints");
        std::vector<IndexT> indices(dofs, 0);
        auto geometry = ExternalArticulationConstraint{}.create_geometry(joints, indices);
        auto mass = view(*geometry["joint_joint"]->find<Float>("mass"));
        Eigen::Map<MatrixX>(mass.data(), dofs, dofs).setIdentity();
        auto object = scene->objects().create("articulation");
        articulation = object->geometries().create(geometry).geometry;
        scene->animator().insert(*object, [](Animation::UpdateInfo&) {});
        world->init(*scene);
        if (!world->is_valid()) throw std::runtime_error("invalid external articulation scene");
    }

    ~Probe() { world.reset(); scene.reset(); engine.reset(); }

    void Step(const double* mass, const double* predicted) {
        const Eigen::Map<const MatrixX> matrix(mass, dofs, dofs);
        const Eigen::Map<const VectorX> delta(predicted, dofs);
        if (!matrix.allFinite() || !delta.allFinite() || !matrix.isApprox(matrix.transpose(), 1.e-10)
                || Eigen::LLT<MatrixX>(matrix).info() != Eigen::Success) {
            throw std::runtime_error("prediction must be finite and mass symmetric positive definite");
        }
        auto& geo = articulation->geometry();
        auto values = view(*geo["joint_joint"]->find<Float>("mass"));
        Eigen::Map<MatrixX>(values.data(), dofs, dofs) = matrix;
        auto prediction = view(*geo["joint"]->find<Float>("delta_theta_tilde"));
        std::copy_n(predicted, dofs, prediction.begin());
        const auto start = std::chrono::steady_clock::now();
        world->advance();
        if (!world->is_valid()) throw std::runtime_error("external articulation step failed");
        world->retrieve();
        const auto diagnostics = world->features().find<SolverDiagnosticsFeature>()->diagnostics();
        Json output;
        output["advance_ms"] = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
        output["newton_iterations"] = diagnostics.newton_iterations;
        output["pcg_relative_residual"] = diagnostics.pcg_relative_residual;
        output["pcg_iterations"] = diagnostics.pcg_iterations_total;
        output["delta"] = geo["joint"]->find<Float>("delta_theta")->view();
        output["positions"] = Json::array();
        for (const auto& vertex : soft->geometry().as<SimplicialComplex>()->positions().view()) {
            output["positions"].push_back({vertex.x(), vertex.y(), vertex.z()});
        }
        output["transforms"] = Json::array();
        for (const auto& slot : moving) {
            const auto& transform = slot->geometry().as<SimplicialComplex>()->transforms().view()[0];
            std::array<double, 16> row{};
            for (int i = 0; i < 16; ++i) row[i] = transform(i / 4, i % 4);
            output["transforms"].push_back(row);
        }
        result = output.dump();
    }
};
}

extern "C" {
const char* gobot_probe_error() { return error.c_str(); }
void* gobot_probe_create(const char* job) {
    try { error.clear(); return new Probe(Json::parse(job)); }
    catch (const std::exception& e) { error = e.what(); return nullptr; }
}
const char* gobot_probe_step(void* handle, const double* mass, const double* prediction) {
    try {
        if (!handle || !mass || !prediction) throw std::runtime_error("null probe argument");
        auto& probe = *static_cast<Probe*>(handle);
        probe.Step(mass, prediction);
        return probe.result.c_str();
    } catch (const std::exception& e) { error = e.what(); return nullptr; }
}
void gobot_probe_destroy(void* handle) { delete static_cast<Probe*>(handle); }
}
