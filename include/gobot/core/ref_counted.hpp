/* Copyright(c) 2020-2022, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Qiqi Wu, 22-11-26
*/

#pragma once

#include "gobot/core/object.hpp"
#include "gobot/core/safe_refcount.hpp"
#include "gobot/core/types.hpp"
#include "gobot/log.hpp"

namespace gobot::core {


class RefCounted : public Object {
    GOBCLASS(RefCounted, Object)
public:
    RefCounted();

    [[nodiscard]] bool IsReferenced() const;

    [[nodiscard]] uint32_t GetReferenceCount() const {
        return refcount_.GetCount();
    }

    bool InitRef();

    // returns false if refcount is at zero and didn't get increased
    bool Reference();

    bool UnReference();

private:
    SafeRefCount refcount_;
    SafeRefCount refcount_init_;
};


template <class T>
class Ref final {
public:
    FORCE_INLINE bool operator==(const T *p_ptr) const {
        return reference_ == p_ptr;
    }
    FORCE_INLINE bool operator!=(const T *p_ptr) const {
        return reference_ != p_ptr;
    }

    FORCE_INLINE bool operator<(const Ref<T> &p_r) const {
        return reference_ < p_r.reference_;
    }
    FORCE_INLINE bool operator==(const Ref<T> &p_r) const {
        return reference_ == p_r.reference_;
    }
    FORCE_INLINE bool operator!=(const Ref<T> &p_r) const {
        return reference_ != p_r.reference_;
    }

    FORCE_INLINE T* operator->() {
        return reference_;
    }

    FORCE_INLINE T* operator*() {
        return reference_;
    }

    FORCE_INLINE const T* operator->() const {
        return reference_;
    }

    FORCE_INLINE const T* GetPtr() const {
        return reference_;
    }
    FORCE_INLINE T* GetPtr() {
        return reference_;
    }

    FORCE_INLINE const T* operator*() const {
        return reference_;
    }

    Ref& operator=(const Ref &from) {
        if (from.reference_ != reference_) {
            unref();

            reference_ = from.reference;
            if (reference_) {
                reference_->reference();
            }
        }
        return *this;
    }

    template <class T_Other>
    Ref& operator=(const Ref<T_Other> &p_from) {
        auto *refb = const_cast<RefCounted *>(static_cast<const RefCounted *>(p_from.GetPtr()));
        if (!refb) {
            unref();
            return *this;
        }
        Ref r;
        r.reference_ = Object::CastTo<T>(refb);
        ref(r);
        r.reference_ = nullptr;
        return *this;
    }

//    void operator=(const Variant &p_variant) {
//        Object *object = p_variant.get_validated_object();
//
//        if (object == reference) {
//            return;
//        }
//
//        unref();
//
//        if (!object) {
//            return;
//        }
//
//        T *r = Object::cast_to<T>(object);
//        if (r && r->reference()) {
//            reference = r;
//        }
//    }

    template <class T_Other>
    void reference_ptr(T_Other *ptr) {
        if (reference_ == ptr) {
            return;
        }
        unref();

        T *r = Object::CastTo<T>(ptr);
        if (r) {
            ref_pointer(r);
        }
    }

    Ref(const Ref &p_from) {
        ref(p_from);
    }

    template <class T_Other>
    explicit Ref(const Ref<T_Other> &from) {
        auto *refb = const_cast<RefCounted *>(static_cast<const RefCounted *>(from.GetPtr()));
        if (!refb) {
            unref();
            return;
        }
        Ref r;
        r.reference_ = Object::CastTo<T>(refb);
        ref(r);
        r.reference_ = nullptr;
    }

    explicit Ref(T *reference) {
        if (reference) {
            ref_pointer(reference);
        }
    }

//    Ref(const Variant &p_variant) {
//        Object *object = p_variant.get_validated_object();
//
//        if (!object) {
//            return;
//        }
//
//        T *r = Object::cast_to<T>(object);
//        if (r && r->reference()) {
//            reference = r;
//        }
//    }

    [[nodiscard]] inline bool IsValid() const { return reference_ != nullptr; }

    [[nodiscard]] inline bool IsNull() const { return reference_ == nullptr; }

    void unref() {
        // TODO: this should be moved to mutexes, since this engine does not really
        // do a lot of referencing on references and stuff
        // mutexes will avoid more crashes?

        if (reference_ && reference_->unreference()) {
            memdelete(reference_);
        }
        reference_ = nullptr;
    }


    void Instantiate() {
        ref(Object::New<T>());
    }

    Ref() = default;

    ~Ref() {
        unref();
    }

private:
    void ref(const Ref &from) {
        if (from.reference_ == reference_) {
            return;
        }

        unref();

        reference_ = from.reference;
        if (reference_) {
            reference_->reference();
        }
    }

    void ref_pointer(T *ref) {
        if (ref == nullptr) {
            LOG_ERROR("The input ref pointer is nullptr");
            return;
        }

        if (ref->init_ref()) {
            reference_ = ref;
        }
    }
private:
    T* reference_{nullptr};
};

}