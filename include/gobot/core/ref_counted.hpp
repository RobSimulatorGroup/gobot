/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-26
*/

#pragma once

#include <rttr/wrapper_mapper.h>

#include "gobot/core/object.hpp"

namespace gobot {


class RefCounted : public Object {
    GOBCLASS(RefCounted, Object)
public:
    ~RefCounted();

    RefCounted();

    RefCounted(const RefCounted &);

    RefCounted &operator=(const RefCounted &);

    void AddRef() noexcept;

    void RemoveRef() noexcept;

    [[nodiscard]] inline bool Unique() const noexcept {
        return rc_ == 1;
    }

    [[nodiscard]] inline size_t GetReferenceCount() const noexcept {
        return rc_;
    }

protected:
    std::atomic<size_t> rc_;
};

template<class T>
class Ref {
    using this_type = Ref<T>;
public:
    using element_type = T;

    Ref() noexcept : ptr_(nullptr) {}

    Ref(T* raw_ptr, bool add_ref = true) noexcept
        : ptr_(raw_ptr)
    {
        if( ptr_ != nullptr && add_ref ) ptr_->AddRef();
    }

    Ref(const Ref &other)
        : ptr_(other.ptr_)
    {
        if(ptr_ != nullptr) ptr_->AddRef();
    }

    template<class U>
    Ref(Ref<U> other) : ptr_(other.Get()) {
        static_assert(std::is_convertible<U*, T*>::value, "Y* is not assignable to T*");
        if(ptr_ != nullptr) ptr_->AddRef();
    }

    ~Ref() {
        if (ptr_ != nullptr) {
            ptr_->RemoveRef();
        }
    }

    template<class U>
    Ref& operator=(const Ref<U>& rhs)
    {
        this_type(rhs).Swap(*this);
        return *this;
    }

    Ref& operator=(const Ref& rhs)
    {
        this_type(rhs).Swap(*this);
        return *this;
    }

    Ref& operator=(T* rhs)
    {
        this_type(rhs).Swap(*this);
        return *this;
    }

    // move semantic
    Ref(Ref && rhs) noexcept
        : ptr_(rhs.ptr_)
    {
        rhs.ptr_ = nullptr;
    }

    Ref& operator=(Ref&& rhs) noexcept
    {
        this_type(static_cast<Ref&&>(rhs)).Swap(*this);
        return *this;
    }

    template<class U>
    Ref& operator=(Ref<U>&& rhs) noexcept
    {
        this_type(static_cast<Ref<U>&&>(rhs)).Swap(*this);
        return *this;
    }


    [[nodiscard]] inline size_t UseCount() const noexcept {
        if (ptr_) ptr_->GetReferenceCount();
        else return 0;
    }

    T* Detach() noexcept {
        T * ret = ptr_;
        ptr_ = nullptr;
        return ret;
    }

    T* Release() noexcept {
        return Detach();
    }

    void Reset()
    {
        this_type().Swap(*this);
    }
    void Reset(T * rhs)
    {
        this_type(rhs).Swap(*this);
    }
    void Reset(T * rhs, bool add_ref )
    {
        this_type(rhs, add_ref).Swap(*this);
    }

    [[nodiscard]] bool IsValid() const noexcept {
        return ptr_ != nullptr;
    }

    T* Get() const noexcept {
        return ptr_;
    }

    T* operator->() const noexcept {
        return ptr_;
    }

    T* operator*() const {
        return *ptr_;
    }

    explicit operator bool() const noexcept {
        return ptr_ != nullptr;
    }

    void Swap(Ref &other) noexcept {
        std::swap(ptr_, other.ptr_);
    }

    template<class C>
    Ref<C> DynamicPointerCast() const noexcept {
        return (ptr_) ? dynamic_cast<C *>(Get()) : nullptr;
    }

    template<class C>
    Ref<C> StaticPointerCast() const noexcept {
        return (ptr_) ? static_cast<C *>(Get()) : nullptr;
    }

    template<typename U>
    constexpr bool operator==(const Ref<U> & r) const noexcept {
        return ptr_ == r.ptr_;
    }
    template<typename U>
    constexpr bool operator==(U * u) const noexcept {
        return ptr_ == u;
    }

    template<typename U>
    friend constexpr bool operator==(U * u, const Ref<T> & r) noexcept {
        return u == r.ptr_;
    }

    template<typename U>
    constexpr bool operator!=(const Ref<U> & r) const noexcept {
        return ptr_ != r.ptr_;
    }
    template<typename U>
    constexpr bool operator!=(U * u) const noexcept {
        return ptr_ != u;
    }
    template<typename U>
    friend constexpr bool operator!=(U * u, const Ref<T>& r) noexcept {
        return u != r.ptr_;
    }

    template<typename U>
    constexpr bool operator<(const Ref<U> & r) const noexcept {
        return ptr_ < r.ptr_;
    }
    template<typename U>
    constexpr bool operator<(U * u) const noexcept {
        return ptr_ < u;
    }
    template<typename U>
    friend constexpr bool operator<(U * u, const Ref<T> & r)  noexcept {
        return u < r.ptr_;
    }

    template<typename U>
    constexpr bool operator>(const Ref<U> & r) const noexcept {
        return ptr_ > r.ptr_;
    }
    template<typename U>
    constexpr bool operator>(U * u) const noexcept {
        return ptr_ > u;
    }
    template<typename U>
    friend constexpr bool operator>(U * u, const Ref<T> & r) noexcept {
        return u > r.ptr_;
    }

    template<typename U>
    constexpr bool operator<=(const Ref<U> & r) const noexcept {
        return ptr_ <= r.ptr_;
    }
    template<typename U>
    constexpr bool operator<=(U * u) const noexcept {
        return ptr_ <= u;
    }
    template<typename U>
    friend constexpr bool operator<=(U * u, const Ref<T> & r) noexcept {
        return u <= r.ptr_;
    }

    template<typename U>
    constexpr bool operator>=(const Ref<U> & r) const noexcept {
        return ptr_ >= r.ptr_;
    }
    template<typename U>
    constexpr bool operator>=(U * u) const noexcept {
        return ptr_ >= u;
    }

    template<typename U>
    friend constexpr bool operator>=(U * u, const Ref<T> & r) noexcept {
        return u >= r.ptr_;
    }

    friend void swap(Ref<T>& l, Ref<T>& r) noexcept {
        l.Swap(r);
    }

private:
    T* ptr_;
};


template<typename T, typename ...Args>
auto MakeRef(Args &&... args){
    static_assert(!std::is_array<T>::value, "Ref does not accept arrays.");
    static_assert(!std::is_reference<T>::value, "Ref does not accept references.");

    return Ref<T>(new T(std::forward<Args>(args)...));
}

template<typename U, typename T>
Ref<U> const_pointer_cast(Ref<T> r) noexcept {
    return const_cast<U*>(r.Get());
}

template<class T, class U>
Ref<T> static_pointer_cast(Ref<U> const &r) noexcept {
return r.template StaticPointerCast<T>();
}

template<class T, class U>
Ref<T> dynamic_pointer_cast(Ref<U> const &r) noexcept {
return r.template DynamicPointerCast<T>();
}


} // end of namespace gobot

namespace rttr {

template<typename T>
struct wrapper_mapper<gobot::Ref<T>> {
    using wrapped_type = decltype(std::declval<gobot::Ref<T>>().Get());
    using type = gobot::Ref<T>;

    static inline wrapped_type get(const type &obj) {
        return obj.Get();
    }

    static RTTR_INLINE gobot::WrapperHolderType get_wrapper_holder_type() {
        return gobot::WrapperHolderType::Ref;
    }

    static inline type create(const wrapped_type &t) {
        return type(t);
    }

    template<typename U>
    static gobot::Ref<U> convert(const type &source, bool &ok) {
        auto cast = source.template DynamicPointerCast<U>();
        if (cast) {
            ok = true;
            return cast;
        } else {
            ok = false;
            return gobot::Ref<U>();
        }
    }
};

}

namespace std {

template<typename T>
struct hash<gobot::Ref<T>>   {
size_t operator()(const gobot::Ref<T>& ptr) const noexcept {
    return std::hash<T*>()(ptr.Get());
}
};

}


