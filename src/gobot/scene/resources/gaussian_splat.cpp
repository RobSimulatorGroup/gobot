/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2026, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/scene/resources/gaussian_splat.hpp"

#include "gobot/core/config/project_setting.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace gobot {
namespace {

enum class PlyFormat {
    Ascii,
    BinaryLittleEndian,
};

enum class PlyScalarType {
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Float32,
    Float64,
};

struct PlyProperty {
    std::string name;
    PlyScalarType type = PlyScalarType::Float32;
    bool is_list = false;
    PlyScalarType list_count_type = PlyScalarType::UInt8;
};

struct PlyElement {
    std::string name;
    std::size_t count = 0;
    std::vector<PlyProperty> properties;
};

struct PlyHeader {
    PlyFormat format = PlyFormat::Ascii;
    std::vector<PlyElement> elements;
};

void SetError(std::string* error, std::string message) {
    if (error != nullptr) {
        *error = std::move(message);
    }
}

std::optional<PlyScalarType> ParseScalarType(std::string_view name) {
    if (name == "char" || name == "int8") return PlyScalarType::Int8;
    if (name == "uchar" || name == "uint8") return PlyScalarType::UInt8;
    if (name == "short" || name == "int16") return PlyScalarType::Int16;
    if (name == "ushort" || name == "uint16") return PlyScalarType::UInt16;
    if (name == "int" || name == "int32") return PlyScalarType::Int32;
    if (name == "uint" || name == "uint32") return PlyScalarType::UInt32;
    if (name == "float" || name == "float32") return PlyScalarType::Float32;
    if (name == "double" || name == "float64") return PlyScalarType::Float64;
    return std::nullopt;
}

bool ParseHeader(std::istream& stream, PlyHeader* header, std::string* error) {
    std::string line;
    if (!std::getline(stream, line) || (line != "ply" && line != "ply\r")) {
        SetError(error, "Gaussian PLY is missing the 'ply' magic header.");
        return false;
    }

    PlyElement* current_element = nullptr;
    bool found_format = false;
    bool found_end = false;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream tokens(line);
        std::string keyword;
        tokens >> keyword;
        if (keyword.empty() || keyword == "comment" || keyword == "obj_info") continue;
        if (keyword == "end_header") {
            found_end = true;
            break;
        }
        if (keyword == "format") {
            std::string format;
            std::string version;
            tokens >> format >> version;
            if (version != "1.0") {
                SetError(error, "Gaussian PLY only supports format version 1.0.");
                return false;
            }
            if (format == "ascii") {
                header->format = PlyFormat::Ascii;
            } else if (format == "binary_little_endian") {
                header->format = PlyFormat::BinaryLittleEndian;
            } else if (format == "binary_big_endian") {
                SetError(error, "Gaussian PLY binary_big_endian data is not supported.");
                return false;
            } else {
                SetError(error, "Gaussian PLY has an unsupported format: " + format + ".");
                return false;
            }
            found_format = true;
            continue;
        }
        if (keyword == "element") {
            PlyElement element;
            tokens >> element.name >> element.count;
            if (!tokens || element.name.empty()) {
                SetError(error, "Gaussian PLY has a malformed element declaration.");
                return false;
            }
            constexpr std::size_t kMaximumElementCount = 100'000'000;
            if (element.count > kMaximumElementCount) {
                SetError(error, "Gaussian PLY element count exceeds the supported limit.");
                return false;
            }
            if (std::any_of(header->elements.begin(), header->elements.end(),
                            [&element](const PlyElement& existing) {
                                return existing.name == element.name;
                            })) {
                SetError(error, "Gaussian PLY contains a duplicate element declaration: " +
                                        element.name + ".");
                return false;
            }
            header->elements.emplace_back(std::move(element));
            current_element = &header->elements.back();
            continue;
        }
        if (keyword == "property") {
            if (current_element == nullptr) {
                SetError(error, "Gaussian PLY property appears before an element declaration.");
                return false;
            }
            std::string type_name;
            tokens >> type_name;
            PlyProperty property;
            if (type_name == "list") {
                std::string count_type;
                std::string value_type;
                tokens >> count_type >> value_type >> property.name;
                const auto parsed_count = ParseScalarType(count_type);
                const auto parsed_value = ParseScalarType(value_type);
                if (!parsed_count || !parsed_value || property.name.empty()) {
                    SetError(error, "Gaussian PLY has a malformed list property.");
                    return false;
                }
                property.is_list = true;
                property.list_count_type = *parsed_count;
                property.type = *parsed_value;
            } else {
                tokens >> property.name;
                const auto parsed_type = ParseScalarType(type_name);
                if (!parsed_type || property.name.empty()) {
                    SetError(error, "Gaussian PLY has an unsupported scalar property type: " + type_name + ".");
                    return false;
                }
                property.type = *parsed_type;
            }
            if (std::any_of(current_element->properties.begin(),
                            current_element->properties.end(),
                            [&property](const PlyProperty& existing) {
                                return existing.name == property.name;
                            })) {
                SetError(error, "Gaussian PLY contains a duplicate property declaration: " +
                                        property.name + ".");
                return false;
            }
            current_element->properties.emplace_back(std::move(property));
            continue;
        }
        SetError(error, "Gaussian PLY has an unsupported header directive: " + keyword + ".");
        return false;
    }

    if (!found_format || !found_end) {
        SetError(error, "Gaussian PLY header is incomplete.");
        return false;
    }
    return true;
}

template <typename T>
bool ReadBinaryValue(std::istream& stream, T* value) {
    std::array<std::uint8_t, sizeof(T)> bytes{};
    stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream) return false;
    if constexpr (std::endian::native == std::endian::big) {
        std::reverse(bytes.begin(), bytes.end());
    }
    std::memcpy(value, bytes.data(), sizeof(T));
    return true;
}

bool ReadScalar(std::istream& stream, PlyFormat format, PlyScalarType type, double* value) {
    if (format == PlyFormat::Ascii) {
        stream >> *value;
        return static_cast<bool>(stream);
    }
    switch (type) {
        case PlyScalarType::Int8: {
            std::int8_t v = 0; if (!ReadBinaryValue(stream, &v)) return false; *value = v; return true;
        }
        case PlyScalarType::UInt8: {
            std::uint8_t v = 0; if (!ReadBinaryValue(stream, &v)) return false; *value = v; return true;
        }
        case PlyScalarType::Int16: {
            std::int16_t v = 0; if (!ReadBinaryValue(stream, &v)) return false; *value = v; return true;
        }
        case PlyScalarType::UInt16: {
            std::uint16_t v = 0; if (!ReadBinaryValue(stream, &v)) return false; *value = v; return true;
        }
        case PlyScalarType::Int32: {
            std::int32_t v = 0; if (!ReadBinaryValue(stream, &v)) return false; *value = v; return true;
        }
        case PlyScalarType::UInt32: {
            std::uint32_t v = 0; if (!ReadBinaryValue(stream, &v)) return false; *value = v; return true;
        }
        case PlyScalarType::Float32: {
            float v = 0.0f; if (!ReadBinaryValue(stream, &v)) return false; *value = v; return true;
        }
        case PlyScalarType::Float64: {
            double v = 0.0; if (!ReadBinaryValue(stream, &v)) return false; *value = v; return true;
        }
    }
    return false;
}

bool SkipProperty(std::istream& stream,
                  PlyFormat format,
                  const PlyProperty& property,
                  std::string* error) {
    if (!property.is_list) {
        double ignored = 0.0;
        return ReadScalar(stream, format, property.type, &ignored);
    }
    double raw_count = 0.0;
    if (!ReadScalar(stream, format, property.list_count_type, &raw_count) ||
        raw_count < 0.0 || raw_count > 100'000'000.0 || std::floor(raw_count) != raw_count) {
        SetError(error, "Gaussian PLY contains an invalid list length.");
        return false;
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(raw_count); ++i) {
        double ignored = 0.0;
        if (!ReadScalar(stream, format, property.type, &ignored)) return false;
    }
    return true;
}

float ActivateOpacity(double value) {
    if (value >= 0.0) {
        return static_cast<float>(1.0 / (1.0 + std::exp(-value)));
    }
    const double e = std::exp(value);
    return static_cast<float>(e / (1.0 + e));
}

bool ReadGaussianVertices(std::istream& stream,
                          const PlyHeader& header,
                          std::shared_ptr<GaussianSplatData>* output,
                          std::string* error) {
    auto vertex_it = std::find_if(header.elements.begin(), header.elements.end(),
                                  [](const PlyElement& element) { return element.name == "vertex"; });
    if (vertex_it == header.elements.end() || vertex_it->count == 0) {
        SetError(error, "Gaussian PLY contains no vertex element.");
        return false;
    }
    for (const PlyProperty& property : vertex_it->properties) {
        if (property.is_list) {
            SetError(error, "Gaussian PLY vertex list properties are not supported.");
            return false;
        }
    }

    std::unordered_map<std::string, std::size_t> property_index;
    for (std::size_t i = 0; i < vertex_it->properties.size(); ++i) {
        property_index.emplace(vertex_it->properties[i].name, i);
    }
    const std::array<const char*, 14> required = {
            "x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3",
            "scale_0", "scale_1", "scale_2", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"};
    for (const char* name : required) {
        if (!property_index.contains(name)) {
            SetError(error, "Gaussian PLY is missing required property '" + std::string(name) + "'.");
            return false;
        }
    }

    std::size_t rest_count = 0;
    while (property_index.contains("f_rest_" + std::to_string(rest_count))) ++rest_count;
    for (const auto& [name, index] : property_index) {
        (void)index;
        if (name.starts_with("f_rest_") && name != "f_rest_" + std::to_string(rest_count)) {
            const std::string suffix = name.substr(7);
            try {
                if (static_cast<std::size_t>(std::stoul(suffix)) >= rest_count) {
                    SetError(error, "Gaussian PLY f_rest properties must be contiguous from f_rest_0.");
                    return false;
                }
            } catch (...) {
                SetError(error, "Gaussian PLY has a malformed f_rest property name.");
                return false;
            }
        }
    }
    int sh_degree = -1;
    for (int degree = 0; degree <= 3; ++degree) {
        const std::size_t coefficients = static_cast<std::size_t>((degree + 1) * (degree + 1));
        if (rest_count == 3u * (coefficients - 1u)) {
            sh_degree = degree;
            break;
        }
    }
    if (sh_degree < 0) {
        SetError(error, "Gaussian PLY SH data must contain 0, 9, 24, or 45 f_rest properties.");
        return false;
    }

    auto data = std::make_shared<GaussianSplatData>();
    data->count = vertex_it->count;
    data->sh_degree = sh_degree;
    const std::size_t coefficient_count = data->GetCoefficientCount();
    data->means.resize(data->count * 3u);
    data->rotations_wxyz.resize(data->count * 4u);
    data->scales.resize(data->count * 3u);
    data->opacities.resize(data->count);
    data->sh_coefficients.resize(data->count * coefficient_count * 3u);

    std::vector<double> values(vertex_it->properties.size());
    bool bounds_initialized = false;
    Vector3 bounds_min = Vector3::Zero();
    Vector3 bounds_max = Vector3::Zero();
    std::size_t vertex_index = 0;
    for (const PlyElement& element : header.elements) {
        for (std::size_t row = 0; row < element.count; ++row) {
            if (element.name != "vertex") {
                for (const PlyProperty& property : element.properties) {
                    if (!SkipProperty(stream, header.format, property, error)) {
                        if (error == nullptr || error->empty()) SetError(error, "Gaussian PLY data ended unexpectedly.");
                        return false;
                    }
                }
                continue;
            }
            for (std::size_t i = 0; i < element.properties.size(); ++i) {
                if (!ReadScalar(stream, header.format, element.properties[i].type, &values[i])) {
                    SetError(error, "Gaussian PLY vertex data ended unexpectedly.");
                    return false;
                }
                if (!std::isfinite(values[i])) {
                    SetError(error, "Gaussian PLY contains a non-finite vertex value.");
                    return false;
                }
            }
            const auto get = [&values, &property_index](const std::string& name) {
                return values[property_index.at(name)];
            };
            const Vector3 mean{
                    static_cast<RealType>(get("x")),
                    static_cast<RealType>(get("y")),
                    static_cast<RealType>(get("z"))};
            data->means[vertex_index * 3u] = static_cast<float>(mean.x());
            data->means[vertex_index * 3u + 1u] = static_cast<float>(mean.y());
            data->means[vertex_index * 3u + 2u] = static_cast<float>(mean.z());

            std::array<double, 4> quaternion = {
                    get("rot_0"), get("rot_1"), get("rot_2"), get("rot_3")};
            const double quaternion_norm = std::sqrt(
                    quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
                    quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);
            if (!(quaternion_norm > 1e-12)) {
                SetError(error, "Gaussian PLY contains a zero-length rotation quaternion.");
                return false;
            }
            for (std::size_t i = 0; i < quaternion.size(); ++i) {
                data->rotations_wxyz[vertex_index * 4u + i] =
                        static_cast<float>(quaternion[i] / quaternion_norm);
            }

            Vector3 scale;
            for (std::size_t i = 0; i < 3; ++i) {
                const double activated = std::exp(get("scale_" + std::to_string(i)));
                if (!std::isfinite(activated) || activated <= 0.0) {
                    SetError(error, "Gaussian PLY contains an invalid activated scale.");
                    return false;
                }
                scale[static_cast<Eigen::Index>(i)] = activated;
                data->scales[vertex_index * 3u + i] = static_cast<float>(activated);
            }
            data->opacities[vertex_index] = ActivateOpacity(get("opacity"));

            const std::size_t sh_base = vertex_index * coefficient_count * 3u;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                data->sh_coefficients[sh_base + channel] =
                        static_cast<float>(get("f_dc_" + std::to_string(channel)));
                for (std::size_t coefficient = 1; coefficient < coefficient_count; ++coefficient) {
                    const std::size_t rest_index = channel * (coefficient_count - 1u) + coefficient - 1u;
                    data->sh_coefficients[sh_base + coefficient * 3u + channel] =
                            static_cast<float>(get("f_rest_" + std::to_string(rest_index)));
                }
            }

            const Vector3 radius = Vector3::Constant(3.0 * scale.maxCoeff());
            const Vector3 point_min = mean - radius;
            const Vector3 point_max = mean + radius;
            if (!bounds_initialized) {
                bounds_min = point_min;
                bounds_max = point_max;
                bounds_initialized = true;
            } else {
                bounds_min = bounds_min.cwiseMin(point_min);
                bounds_max = bounds_max.cwiseMax(point_max);
            }
            ++vertex_index;
        }
    }
    data->bounds = AABB::FromMinMax(bounds_min, bounds_max);
    if (!data->IsValid()) {
        SetError(error, "Gaussian PLY produced inconsistent Gaussian arrays.");
        return false;
    }
    *output = std::move(data);
    return true;
}

} // namespace

bool GaussianSplatData::IsValid() const {
    const std::size_t coefficient_count = GetCoefficientCount();
    return count > 0 && sh_degree >= 0 && sh_degree <= 3 &&
           means.size() == count * 3u && rotations_wxyz.size() == count * 4u &&
           scales.size() == count * 3u && opacities.size() == count &&
           sh_coefficients.size() == count * coefficient_count * 3u && bounds.IsValid();
}

void GaussianSplatResource::SetSourcePath(const std::string& path) {
    if (source_path_ == path) return;
    if (path.empty()) {
        source_path_.clear();
        data_.reset();
        MarkChanged();
        return;
    }
    const std::string input_path = path.starts_with("res://") && ProjectSettings::HasInstance()
                                           ? ProjectSettings::GetInstance()->GlobalizePath(path)
                                           : path;
    if (data_ != nullptr &&
        std::filesystem::path(input_path).lexically_normal() ==
                std::filesystem::path(source_path_).lexically_normal()) {
        source_path_ = path;
        MarkChanged();
        return;
    }
    std::string error;
    if (!LoadPly(input_path, &error)) {
        source_path_ = path;
        data_.reset();
        LOG_ERROR("Failed to set Gaussian Splatting source '{}': {}", path, error);
        MarkChanged();
        return;
    }
    source_path_ = path;
    MarkChanged();
}

const std::string& GaussianSplatResource::GetSourcePath() const {
    return source_path_;
}

std::size_t GaussianSplatResource::GetGaussianCount() const {
    return data_ != nullptr ? data_->count : 0u;
}

int GaussianSplatResource::GetShDegree() const {
    return data_ != nullptr ? data_->sh_degree : 0;
}

AABB GaussianSplatResource::GetBounds() const {
    return data_ != nullptr ? data_->bounds : AABB{};
}

std::shared_ptr<const GaussianSplatData> GaussianSplatResource::GetData() const {
    return data_;
}

bool GaussianSplatResource::LoadPly(const std::string& path, std::string* error) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        SetError(error, "Cannot open Gaussian PLY file: " + path + ".");
        return false;
    }
    PlyHeader header;
    if (!ParseHeader(stream, &header, error)) return false;
    std::shared_ptr<GaussianSplatData> parsed;
    if (!ReadGaussianVertices(stream, header, &parsed, error)) return false;
    source_path_ = path;
    data_ = std::move(parsed);
    MarkChanged();
    return true;
}

void GaussianSplatResource::ResetState() {
    source_path_.clear();
    data_.reset();
    MarkChanged();
}

}

GOBOT_REGISTRATION {
    USING_ENUM_BITWISE_OPERATORS;

    Type::register_wrapper_converter_for_base_classes<
            Ref<GaussianSplatResource>, Ref<Resource>>();

    Class_<GaussianSplatResource>("GaussianSplatResource")
            .constructor()(CtorAsRawPtr)
            .property("source_path",
                      &GaussianSplatResource::GetSourcePath,
                      &GaussianSplatResource::SetSourcePath)(
                    AddMetaPropertyInfo(PropertyInfo().SetUsageFlags(
                            PropertyUsageFlags::Storage | PropertyUsageFlags::Editor)));
}
