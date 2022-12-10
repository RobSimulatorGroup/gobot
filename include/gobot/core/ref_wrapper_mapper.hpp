/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-10
*/

#pragma once

#include <rttr/wrapper_mapper.h>

namespace rttr {

template<typename T>
struct wrapper_mapper<gobot::Ref<T>> {
    using wrapped_type  = decltype(std::declval<gobot::Ref<T>>().get());
    using type = gobot::Ref<T>;

    static inline wrapped_type get(const type& obj)
    {
        return obj.get();
    }

    static RTTR_INLINE rttr::wrapper_holder_type get_wrapper_holder_type()
    {
        return rttr::wrapper_holder_type::Ref;
    }

    static inline type create(const wrapped_type& t)
    {
        return type(t);
    }

    template<typename U>
    static gobot::Ref<U> convert(const type& source, bool& ok)
    {
        if (auto p = rttr_cast<typename gobot::Ref<U>::element_type*>(source.get()))
        {
            ok = true;
            return gobot::Ref<U>(source, p);
        }
        else
        {
            ok = false;
            return gobot::Ref<U>();
        }
    }
};

}

