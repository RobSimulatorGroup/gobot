#include "gobot/python/python_binding_registry.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

#include <pybind11/stl.h>

namespace gobot::python {
namespace {

py::object ArithmeticVariantToPython(const Variant& variant) {
    const Type type = variant.get_type();
    if (type == Type::get<bool>()) {
        return py::bool_(variant.to_bool());
    }
    if (type == Type::get<std::uint8_t>() ||
        type == Type::get<std::uint16_t>() ||
        type == Type::get<std::uint32_t>() ||
        type == Type::get<std::uint64_t>() ||
        type == Type::get<std::int8_t>() ||
        type == Type::get<std::int16_t>() ||
        type == Type::get<std::int32_t>() ||
        type == Type::get<std::int64_t>() ||
        type == Type::get<int>() ||
        type == Type::get<unsigned int>()) {
        return py::int_(variant.to_int64());
    }
    if (type == Type::get<float>() ||
        type == Type::get<double>() ||
        type == Type::get<RealType>()) {
        return py::float_(variant.to_double());
    }
    return py::none();
}

py::object SequentialVariantToPython(Variant variant) {
    VariantListView view = variant.create_sequential_view();
    py::list list;
    for (const Variant& value : view) {
        list.append(VariantToPython(value.extract_wrapped_value()));
    }
    return list;
}

Variant NumericPythonToVariantForType(const py::handle& object, const Type& type) {
    if (type == Type::get<float>()) {
        return Variant(py::cast<float>(object));
    }
    if (type == Type::get<double>() ||
        type == Type::get<RealType>()) {
        return Variant(py::cast<RealType>(object));
    }
    if (type == Type::get<std::uint8_t>()) {
        return Variant(py::cast<std::uint8_t>(object));
    }
    if (type == Type::get<std::uint16_t>()) {
        return Variant(py::cast<std::uint16_t>(object));
    }
    if (type == Type::get<std::uint32_t>()) {
        return Variant(py::cast<std::uint32_t>(object));
    }
    if (type == Type::get<std::uint64_t>()) {
        return Variant(py::cast<std::uint64_t>(object));
    }
    if (type == Type::get<std::int8_t>()) {
        return Variant(py::cast<std::int8_t>(object));
    }
    if (type == Type::get<std::int16_t>()) {
        return Variant(py::cast<std::int16_t>(object));
    }
    if (type == Type::get<std::int32_t>() || type == Type::get<int>()) {
        return Variant(py::cast<std::int32_t>(object));
    }
    if (type == Type::get<std::int64_t>()) {
        return Variant(py::cast<std::int64_t>(object));
    }
    return {};
}

bool FillSequentialViewFromPython(VariantListView& view, const py::handle& object) {
    py::sequence sequence = py::reinterpret_borrow<py::sequence>(object);
    if (!view.set_size(sequence.size())) {
        return false;
    }

    const Type value_type = view.get_value_type();
    for (std::size_t index = 0; index < static_cast<std::size_t>(sequence.size()); ++index) {
        if (value_type.is_sequential_container()) {
            Variant nested_value = view.get_value(index);
            VariantListView nested_view = nested_value.create_sequential_view();
            if (!FillSequentialViewFromPython(nested_view, sequence[index]) ||
                !view.set_value(index, nested_value)) {
                return false;
            }
            continue;
        }

        Variant item_value = PythonToVariantForType(sequence[index], value_type);
        if (!item_value.is_valid() || !view.set_value(index, item_value)) {
            return false;
        }
    }
    return true;
}

Variant SequenceToVariantForType(const py::handle& object, const Type& type) {
    Variant value = type.create();
    if (!value.is_valid() || !value.is_sequential_container()) {
        return {};
    }

    VariantListView view = value.create_sequential_view();
    if (!FillSequentialViewFromPython(view, object)) {
        return {};
    }
    return value;
}

} // namespace

py::tuple Vector3ToPython(const Vector3& vector) {
    return py::make_tuple(vector.x(), vector.y(), vector.z());
}

Vector3 PythonToVector3(const py::handle& object) {
    py::sequence sequence = py::reinterpret_borrow<py::sequence>(object);
    if (sequence.size() != 3) {
        throw std::invalid_argument("expected a 3-element vector");
    }
    return {
            py::cast<RealType>(sequence[0]),
            py::cast<RealType>(sequence[1]),
            py::cast<RealType>(sequence[2])
    };
}

py::object VariantToPython(const Variant& variant) {
    if (!variant.is_valid()) {
        return py::none();
    }

    Variant value = variant;
    if (value.get_type().is_wrapper()) {
        value = value.extract_wrapped_value();
    }

    const Type type = value.get_type();
    if (type == Type::get<std::string>()) {
        return py::str(value.to_string());
    }
    if (type == Type::get<Vector3>()) {
        return Vector3ToPython(value.get_value<Vector3>());
    }
    if (type.is_enumeration()) {
        const py::object py_type = py::module_::import("gobot").attr(type.get_name().data());
        return py_type(value.to_int());
    }
    if (type.is_arithmetic()) {
        return ArithmeticVariantToPython(value);
    }
    if (value.is_sequential_container()) {
        return SequentialVariantToPython(value);
    }
    if (!type.get_properties().empty()) {
        return ReflectedToPythonDict(value);
    }

    return py::str(value.to_string());
}

Variant PythonToVariant(const py::handle& object) {
    if (object.is_none()) {
        return {};
    }
    if (py::isinstance<py::bool_>(object)) {
        return Variant(py::cast<bool>(object));
    }
    if (py::isinstance<py::int_>(object)) {
        return Variant(py::cast<std::int64_t>(object));
    }
    if (py::isinstance<py::float_>(object)) {
        return Variant(py::cast<RealType>(object));
    }
    if (py::isinstance<py::str>(object)) {
        return Variant(py::cast<std::string>(object));
    }
    if (py::isinstance<py::dict>(object)) {
        throw std::invalid_argument("cannot infer Gobot reflected type from a Python dict");
    }
    throw std::invalid_argument("unsupported Python object for Gobot Variant conversion");
}

Variant PythonToVariantForType(const py::handle& object, const Type& type) {
    if (type == Type::get<bool>()) {
        return Variant(py::cast<bool>(object));
    }
    if (type.is_arithmetic()) {
        Variant value = NumericPythonToVariantForType(object, type);
        if (value.is_valid()) {
            return value;
        }
    }
    if (type == Type::get<std::string>()) {
        return Variant(py::cast<std::string>(object));
    }
    if (type == Type::get<Vector3>()) {
        return Variant(PythonToVector3(object));
    }
    if (type.is_enumeration()) {
        if (py::isinstance<py::str>(object)) {
            Variant value = type.get_enumeration().name_to_value(py::cast<std::string>(object));
            if (!value.is_valid()) {
                throw std::invalid_argument("unknown enum value '" + py::cast<std::string>(object) +
                                            "' for " + type.get_name().to_string());
            }
            return value;
        }
        const std::string name = py::cast<std::string>(object.attr("name"));
        return type.get_enumeration().name_to_value(name);
    }
    if (type.is_sequential_container()) {
        Variant value = SequenceToVariantForType(object, type);
        if (value.is_valid()) {
            return value;
        }
    }
    if (py::isinstance<py::dict>(object) && !type.get_properties().empty()) {
        return DictToReflected(py::reinterpret_borrow<py::dict>(object), type);
    }
    throw std::invalid_argument("unsupported Gobot reflected property type '" +
                                type.get_name().to_string() + "'");
}

bool SetReflectedPropertyFromPython(Instance instance, const Property& property, const py::handle& value) {
    if (property.get_type().is_sequential_container()) {
        Variant container = property.get_value(instance);
        VariantListView view = container.create_sequential_view();
        return FillSequentialViewFromPython(view, value) && property.set_value(instance, container);
    }

    Variant property_value = PythonToVariantForType(value, property.get_type());
    return property_value.is_valid() && property.set_value(instance, property_value);
}

py::dict ReflectedToPythonDict(const Variant& reflected_value) {
    Variant value = reflected_value;
    if (value.get_type().is_wrapper()) {
        value = value.extract_wrapped_value();
    }

    const Type type = value.get_type();
    py::dict dict;
    for (const Property& property : type.get_properties()) {
        Variant property_value = property.get_value(value);
        if (!property_value.is_valid()) {
            continue;
        }
        dict[py::str(property.get_name().data())] = VariantToPython(property_value);
    }
    return dict;
}

Variant DictToReflected(py::dict dict, const Type& type) {
    Variant value = type.create();
    if (!value.is_valid()) {
        throw std::invalid_argument("cannot default-construct reflected Gobot type '" +
                                    type.get_name().to_string() + "'");
    }

    for (const auto& item : dict) {
        const std::string name = py::cast<std::string>(item.first);
        const Property property = type.get_property(name);
        if (!property.is_valid()) {
            throw std::invalid_argument("unknown property '" + name + "' for " + type.get_name().to_string());
        }

        if (!SetReflectedPropertyFromPython(value, property, item.second)) {
            throw std::invalid_argument("failed to set property '" + name + "' for " +
                                        type.get_name().to_string());
        }
    }
    return value;
}

} // namespace gobot::python
