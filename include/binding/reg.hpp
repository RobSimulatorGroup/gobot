
#pragma once


#ifdef BUILD_WITH_PYBIND11
#include <Python.h>
#include <pybind11/pybind11.h>
#endif


#include "gobot/core/registration.hpp"

namespace gobot {

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

    template<typename... Args, typename AccLevel = PublicAccess,
            typename Tp = typename std::enable_if<rttr::detail::contains<AccLevel, rttr::detail::access_levels_list>::value>::type>
    ClassR_& Constructor(AccLevel level = AccLevel()) {
#ifdef BUILD_WITH_PYBIND11
        class_obj_.def(pybind11::init<Args...>());
#else
        class_obj_.template constructor<Args...>();
#endif
        return *this;
    }

    // This used for virtual class.
    template<typename Func ,
            typename AccLevel = PublicAccess,
            typename Tp = typename std::enable_if<!rttr::detail::contains<Func, rttr::detail::access_levels_list>::value>::type>
    ClassR_& Constructor(Func&& func, AccLevel acc_level = AccLevel()) {
#ifdef BUILD_WITH_PYBIND11
        // pybind11 handle virtual class using trampoline class
#else
        class_obj_.template constructor(std::forward<Func>(func));
#endif
    }

    template<typename... Args>
    ClassR_& operator()	(Args &&... args) {
#ifdef BUILD_WITH_PYBIND11
#else
        class_obj_.operator()(std::forward<Args...>(args...));
#endif
        return *this;
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

    template<typename A1, typename A2,
            typename AccLevel = PublicAccess,
            typename Tp = typename std::enable_if<!rttr::detail::contains<A2, rttr::detail::access_levels_list>::value>::type>
    ClassR_& Property(const char* name, A1 getter, A2 setter, AccLevel level = AccLevel()) {
#ifdef BUILD_WITH_PYBIND11
        class_obj_.def_property(name, getter, setter);
#else
        class_obj_.property(name, getter, setter, level);
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