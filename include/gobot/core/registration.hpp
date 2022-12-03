/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-6
*/

#pragma once

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

}

#define GOBOT_REGISTRATION RTTR_REGISTRATION
