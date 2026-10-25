#include "manual_bindings_internal.hpp"
#include "gobot/simulation/tactile_sensor_processor.hpp"
#include <cstring>

namespace gobot::python {
namespace {
Affine3 Matrix(const py::handle &values) {
    auto a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(values);
    if (!a || a.size() != 16)
        throw py::value_error("expected a 4x4 tactile transform");
    Affine3 result;
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            result.matrix()(row, col) = a.data()[row * 4 + col];
    return result;
}
template <class T> py::array_t<T> Array(const std::vector<T> &values, std::vector<py::ssize_t> shape) {
    py::array_t<T> array(shape);
    if (std::size_t(array.size()) != values.size())
        throw std::logic_error("tactile output shape mismatch");
    if (!values.empty())
        std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(T));
    array.attr("setflags")(false);
    return array;
}
template <int N, class V> py::array_t<double> Vectors(const std::vector<V> &values) {
    py::array_t<double> array({py::ssize_t(values.size()), py::ssize_t(N)});
    for (std::size_t i = 0; i < values.size(); ++i)
        for (int j = 0; j < N; ++j)
            array.mutable_data()[i * N + j] = values[i][j];
    array.attr("setflags")(false);
    return array;
}
py::dict ToPython(const TactileSensorFrame &f) {
    py::dict d;
    const py::ssize_t w = f.width, h = f.height;
    d["rgb"] = Array(f.rgb, {h, w, 3});
    d["rgb_marker"] = Array(f.rgb_marker, {h, w, 3});
    d["contact_depth"] = Array(f.contact_depth, {h, w});
    d["optical_indentation"] = Array(f.optical_indentation, {h, w});
    d["depth"] = Array(f.depth, {h, w});
    d["indentation"] = Array(f.indentation, {h, w});
    d["normal_pressure"] = Array(f.normal_pressure, {h, w});
    d["shear_stress"] = Array(f.shear_stress, {h, w, 2});
    d["marker_positions"] = Vectors<2>(f.marker_positions);
    d["marker_motion"] = Vectors<2>(f.marker_motion);
    d["local_vertices"] = Vectors<3>(f.local_vertices);
    d["total_force"] =
        Array(std::vector<double>{f.total_force.x(), f.total_force.y(), f.total_force.z()}, {3});
    d["clock"] = Array(std::vector<double>{double(f.tick), f.simulation_time}, {2});
    return d;
}
class PyTactileProcessor {
  public:
    PyTactileProcessor(const py::dict &sensor, const py::handle &vertices,
                       const std::vector<std::uint32_t> &tetrahedra,
                       const std::vector<std::uint32_t> &surface) {
        const auto j =
            nlohmann::json::parse(py::module_::import("json").attr("dumps")(sensor).cast<std::string>());
        TactileSensorParameters p;
        p.image_height = j.at("resolution")[0];
        p.image_width = j.at("resolution")[1];
        p.near_plane = j.at("near_plane");
        p.far_plane = j.at("far_plane");
        p.pixel_size = j.at("pixel_size");
        p.rgb_model = j.at("rgb_model");
        p.coat_vertex_indices = j.at("coat_vertex_indices").get<std::vector<std::uint32_t>>();
        p.stick_vertex_indices = j.at("stick_vertex_indices").get<std::vector<std::uint32_t>>();
        p.marker_tetrahedra = j.at("marker_tetrahedra").get<std::vector<std::uint32_t>>();
        for (const auto &v : j.at("marker_positions"))
            p.marker_positions.push_back(Vector2{v[0].get<RealType>(), v[1].get<RealType>()});
        for (const auto &v : j.at("marker_barycentric"))
            p.marker_barycentric.push_back(Vector4{v[0].get<RealType>(), v[1].get<RealType>(),
                                                   v[2].get<RealType>(), v[3].get<RealType>()});
        if (sensor.contains("camera_intrinsics")) {
            const auto v = j.at("camera_intrinsics");
            p.camera_intrinsics = Vector4{v[0].get<RealType>(), v[1].get<RealType>(), v[2].get<RealType>(),
                                          v[3].get<RealType>()};
            p.camera_transform = Matrix(sensor["camera_transform"]["matrix_row_major"]);
        }
        TactileOpticalData calibration;
        if (j.contains("optical_calibration")) {
            const auto &c = j.at("optical_calibration");
            calibration.pixel_size = c.value("pixel_size", RealType(2.95e-5));
            calibration.width = c.at("width");
            calibration.height = c.at("height");
            calibration.bins = c.at("bins");
            calibration.calibration_width = c.at("calibration_width");
            calibration.calibration_height = c.at("calibration_height");
            calibration.coefficients = c.at("coefficients").get<std::vector<RealType>>();
            calibration.background = c.at("background").get<std::vector<RealType>>();
        }
        processor_ = std::make_unique<TactileSensorProcessor>(std::move(p), PythonToVector3List(vertices),
                                                              tetrahedra, surface, std::move(calibration));
    }
    py::dict Process(const py::handle &positions, const py::handle &forces, const py::handle &transform,
                     std::uint64_t tick, double time, const py::handle &contact_vertices,
                     const std::vector<std::uint32_t> &contact_triangles) {
        auto points = PythonToVector3List(positions), contact = PythonToVector3List(forces);
        const auto pose = Matrix(transform);
        auto colliders =
            contact_vertices.is_none() ? std::vector<Vector3>{} : PythonToVector3List(contact_vertices);
        std::shared_ptr<const TactileSensorFrame> frame;
        {
            py::gil_scoped_release release;
            frame = processor_->Process(points, contact, pose, tick, time, colliders, contact_triangles);
        }
        return ToPython(*frame);
    }

  private:
    std::unique_ptr<TactileSensorProcessor> processor_;
};
template <class T> std::vector<T> Read(const py::dict &d, const char *name, std::vector<py::ssize_t> shape) {
    auto a = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(d[name]);
    if (!a || a.ndim() != py::ssize_t(shape.size()))
        throw py::value_error(std::string("invalid tactile field ") + name);
    for (std::size_t i = 0; i < shape.size(); ++i)
        if (a.shape(i) != shape[i])
            throw py::value_error(std::string("invalid tactile field shape ") + name);
    for (py::ssize_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(double(a.data()[i])))
            throw py::value_error("non-finite tactile frame");
    return {a.data(), a.data() + a.size()};
}
} // namespace
void RegisterManualTactileBindings(py::module_ &module) {
    py::class_<PyTactileProcessor>(module, "_TactileSensorProcessor")
        .def(py::init<const py::dict &, const py::handle &, const std::vector<std::uint32_t> &,
                      const std::vector<std::uint32_t> &>())
        .def("process", &PyTactileProcessor::Process, py::arg("positions"), py::arg("forces"),
             py::arg("transform"), py::arg("tick"), py::arg("time"), py::arg("contact_vertices") = py::none(),
             py::arg("contact_triangles") = std::vector<std::uint32_t>{});
    // Presentation is an engine operation on the runtime node, not an ImGui binding.
    auto cls = module.attr("TactileSensor3D");
    cls.attr("_apply_runtime_frame") = py::cpp_function(
        [](PyTactileSensor3DHandle &handle, const py::dict &d) {
            auto *sensor = handle.ResolveAs<TactileSensor3D>();
            const auto config = sensor->GetConfig();
            if (!config.IsValid() || !sensor->GetGelMesh().IsValid())
                throw py::value_error("tactile node has no config or mesh");
            auto f = std::make_shared<TactileSensorFrame>();
            const py::ssize_t w = config->GetImageWidth(), h = config->GetImageHeight();
            f->width = w;
            f->height = h;
            f->rgb = Read<std::uint8_t>(d, "rgb", {h, w, 3});
            f->rgb_marker = Read<std::uint8_t>(d, "rgb_marker", {h, w, 3});
            f->contact_depth = Read<float>(d, "contact_depth", {h, w});
            f->optical_indentation = Read<float>(d, "optical_indentation", {h, w});
            f->depth = Read<float>(d, "depth", {h, w});
            f->indentation = Read<float>(d, "indentation", {h, w});
            f->normal_pressure = Read<float>(d, "normal_pressure", {h, w});
            f->shear_stress = Read<float>(d, "shear_stress", {h, w, 2});
            const auto m = config->GetMarkerPositions().size();
            auto positions = Read<double>(d, "marker_positions", {py::ssize_t(m), 2});
            auto motions = Read<double>(d, "marker_motion", {py::ssize_t(m), 2});
            for (std::size_t i = 0; i < m; ++i) {
                f->marker_positions.emplace_back(positions[2 * i], positions[2 * i + 1]);
                f->marker_motion.emplace_back(motions[2 * i], motions[2 * i + 1]);
            }
            f->local_vertices = PythonToVector3List(d["local_vertices"]);
            if (f->local_vertices.size() != sensor->GetGelMesh()->GetVertexCount())
                throw py::value_error("tactile frame topology mismatch");
            auto force = Read<double>(d, "total_force", {3});
            f->total_force = Vector3{RealType(force[0]), RealType(force[1]), RealType(force[2])};
            auto clock = Read<double>(d, "clock", {2});
            f->tick = std::uint64_t(clock[0]);
            f->simulation_time = clock[1];
            sensor->SetRuntimeFrame(std::move(f));
        },
        py::is_method(cls));
}
} // namespace gobot::python
