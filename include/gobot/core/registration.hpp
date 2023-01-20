/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-6
*/

#pragma once

#ifdef BUILD_WITH_PYBIND11
#include <Python.h>
#include <pybind11/pybind11.h>
#endif

#include <rttr/registration>

#include <magic_enum.hpp>


namespace gobot {

using PublicAccess = rttr::detail::public_access;
using ProtectedAccess = rttr::detail::protected_access;
using PrivateAccess = rttr::detail::private_access;

namespace CtorPolicy {
using AsSharedPtr = rttr::detail::as_std_shared_ptr; // The default constructor
using AsObject = rttr::detail::as_object;
using AsRawPtr = rttr::detail::as_raw_pointer;
}

static const CtorPolicy::AsSharedPtr CtorAsSharedPtr; // The default constructor
static const CtorPolicy::AsObject CtorAsObject; // The default constructor
static const CtorPolicy::AsRawPtr CtorAsRawPtr; // The default constructor




// The default constructor create policy is: CtorAsSharedPtr
template <typename T>
using Class_ = rttr::registration::class_<T>;

template<typename Enum>
auto Enumeration_ = rttr::registration::enumeration<Enum>;

template<typename Enum, std::size_t size, std::size_t... I>
inline void EnumRegistrationImpl(std::string_view name,
                                 const std::array<rttr::detail::enum_data<Enum>, size>& array,
                                 std::index_sequence<I...>) {
    auto enum_binder = Enumeration_<Enum>(name.data());
    enum_binder(rttr::detail::enum_data<Enum>(array[I])...);
}

template<typename Enum>
inline void QuickEnumeration_(std::string_view name) {
    constexpr auto array = magic_enum::enum_values<Enum>();
    constexpr auto size = magic_enum::enum_count<Enum>();
    std::array<rttr::detail::enum_data<Enum>, size> enum_data_map;
    for (std::size_t i = 0; i < size; i++) {
        enum_data_map[i] = rttr::value(magic_enum::enum_name(array[i]).data(), array[i]);
    }
    EnumRegistrationImpl(name, enum_data_map, std::make_index_sequence<size>());
}


// This is copy from pybind11
template <typename... Args>
struct overload_cast_impl {
    template <typename Return>
    constexpr auto operator()(Return (*pf)(Args...)) const noexcept -> decltype(pf) {
        return pf;
    }

    template <typename Return, typename Class>
    constexpr auto operator()(Return (Class::*pmf)(Args...), std::false_type = {}) const noexcept
    -> decltype(pmf) {
        return pmf;
    }

    template <typename Return, typename Class>
    constexpr auto operator()(Return (Class::*pmf)(Args...) const, std::true_type) const noexcept
    -> decltype(pmf) {
        return pmf;
    }
};

/// Syntax sugar for resolving overloaded function pointers:
///  - regular: static_cast<Return (Class::*)(Arg0, Arg1, Arg2)>(&Class::func)
///  - sweet:   overload_cast<Arg0, Arg1, Arg2>(&Class::func)
template <typename... Args>
static constexpr overload_cast_impl<Args...> overload_cast{};
/// Const member function selector for overload_cast
///  - regular: static_cast<Return (Class::*)(Arg) const>(&Class::func)
///  - sweet:   overload_cast<Arg>(&Class::func, const_)

namespace detail {
static constexpr auto const_ = std::true_type{};
}

}

#define GOBOT_REGISTRATION                                                          \
namespace gobot {                                                                   \
    static void gobot_auto_register_reflection_function_();                         \
}                                                                                   \
namespace                                                                           \
{                                                                                   \
    struct gobot__auto__register__                                                  \
    {                                                                               \
        gobot__auto__register__()                                                   \
        {                                                                           \
            gobot::gobot_auto_register_reflection_function_();                      \
        }                                                                           \
    };                                                                              \
}                                                                                   \
static const gobot__auto__register__ RTTR_CAT(auto_register__, __LINE__);           \
static void gobot::gobot_auto_register_reflection_function_()


namespace gobot {

enum class CtorPolicyType {
    AsSharedPtr,
    AsObject,
    AsRawPtr
};

struct DummyModule{};

template<typename ClassType, typename... Options>
class ClassR_ {
public:

#ifdef BUILD_WITH_PYBIND11
    ClassR_(::pybind11::module_& m, const char* name)
        : class_obj_(m, name)
#else
    ClassR_(void*, const char* name)
        : class_obj_(name)
#endif
    {
    }

    template<typename... Args, typename AccLevel = gobot::PublicAccess,
             typename Tp = typename std::enable_if<rttr::detail::contains<AccLevel, rttr::detail::access_levels_list>::value>::type>
    ClassR_& Constructor(CtorPolicyType ctor_policy_type, AccLevel level = AccLevel()) {
#ifdef BUILD_WITH_PYBIND11
        class_obj_.def(pybind11::init<Args...>());
#else
        switch (ctor_policy_type) {
            case CtorPolicyType::AsSharedPtr:
                class_obj_.template constructor<Args...>(level)(CtorAsSharedPtr);
                break;
            case CtorPolicyType::AsObject:
                class_obj_.template constructor<Args...>(level)(CtorAsObject);
                break;
            case CtorPolicyType::AsRawPtr:
                class_obj_.template constructor<Args...>(level)(CtorAsRawPtr);
                break;
        }
#endif
        return *this;
    }

    // This used for virtual class.
    template<typename Func ,
             typename AccLevel = PublicAccess,
             typename Tp = typename std::enable_if<!rttr::detail::contains<Func, rttr::detail::access_levels_list>::value>::type>
    ClassR_& Constructor(Func&& func, CtorPolicyType ctor_policy_type, AccLevel acc_level = AccLevel()) {
#ifdef BUILD_WITH_PYBIND11
        // pybind11 handle virtual class using trampoline class
#else
        class_obj_.template constructor(acc_level)(ctor_policy_type);
#endif

    }

    template<typename A,
             typename AccLevel = PublicAccess,
             typename Tp = typename std::enable_if<rttr::detail::contains<AccLevel, rttr::detail::access_levels_list>::value>::type>
    ClassR_& Property(const char* name, A acc, AccLevel level = AccLevel()) {
#ifdef BUILD_WITH_PYBIND11
        class_obj_.def_readwrite(name, acc);
#else
        class_obj_.property(name, acc, level);
#endif
        return *this;
    }

private:
#ifdef BUILD_WITH_PYBIND11
    pybind11::class_<ClassType, Options...> class_obj_;
#else
    rttr::registration::class_<ClassType> class_obj_;
#endif

    const char* name_{};

};


}