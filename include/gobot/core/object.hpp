/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-9-24
*/

#pragma once

#include <rttr/rttr_enable.h>
#include <QObject>

#include "gobot/core/types.hpp"
#include "gobot/core/macros.hpp"
#include "gobot/core/notification_enum.hpp"
#include "gobot/core/spin_lock.hpp"
#include "gobot/core/object_id.hpp"


// Do nothing for 1
#define GOBOT_CLASS_1(...)    \


#define GOBOT_CLASS_2(Class, BaseClass)                                                             \
protected:                                                                                          \
    static FORCE_INLINE auto GetNotificationCallBack() -> void(Object::*)(NotificationType) {                    \
         return (void(Object::*)(NotificationType)) &Class::NotificationCallBack;                                \
    }                                                                                               \
    void NotificationImpl(NotificationType notification, bool reversed) override {                  \
        if (!reversed) { BaseClass::NotificationImpl(notification, reversed); }                     \
                                                                                                    \
	    if (Class::GetNotificationCallBack() != BaseClass::GetNotificationCallBack()) {             \
            NotificationCallBack(notification);                                                             \
		}                                                                                           \
		if (reversed) { BaseClass::NotificationImpl(notification, reversed); }                      \
    }

// Unsupported usage.
#define GOBOT_CLASS_3(...) static_assert(false, "GOBCLASS can only have 1 or 2 parameter, but you give 3!");
#define GOBOT_CLASS_4(...) static_assert(false, "GOBCLASS can only have 1 or 2 parameter, but you give 4!");
#define GOBOT_CLASS_5(...) static_assert(false, "GOBCLASS can only have 1 or 2 parameter, but you give 5!");
#define GOBOT_CLASS_6(...) static_assert(false, "GOBCLASS can only have 1 or 2 parameter, but you give 6!");


#define GOBOT_CLASS_PARAM(x) CAT2(GOBOT_CLASS_, x)

#define GOBOT_CLASS_TO_RTTR(A, ...) RTTR_ENABLE(__VA_ARGS__)


#define GOBCLASS(...)                                             \
    GOBOT_CLASS_PARAM(PP_NARG_COUNT(__VA_ARGS__))(__VA_ARGS__)    \
    GOBOT_CLASS_TO_RTTR(__VA_ARGS__)


namespace gobot {


enum class PropertyHint {
    None, ///< no hint provided.
    Range, ///< hint_text = "min,max[,step][,or_greater][,or_less][,hide_slider][,radians][,degrees][,exp][,suffix:<keyword>] range.
    Flags, ///< hint_text= "flag1,flag2,etc" (as bit flags)
    File, ///< a file path must be passed, hint_text (optionally) is a filter "*.png,*.wav,*.doc,"
    Dir, ///< a directory path must be passed
    GlobalFile, ///< a file path must be passed, hint_text (optionally) is a filter "*.png,*.wav,*.doc,"
    GlobalDir, ///< a directory path must be passed
};

enum class PropertyUsageFlags {
    None = 0,
    Storage = 1 << 1,
    Editor = 1 << 2,
    NotSharedOnClone = 1 << 3,
    Default = Storage | Editor
};



struct PropertyInfo {
    String name;
    PropertyHint hint = PropertyHint::None;
    String hint_string;
    PropertyUsageFlags usage = PropertyUsageFlags::Default;

    PropertyInfo& SetName(const String& _name) {
        name = _name;
        return *this;
    }

    PropertyInfo& SetHint(const PropertyHint& property_hint) {
        hint = property_hint;
        return *this;
    }

    PropertyInfo& SetHintString(const String& _hint_string) {
        hint_string = _hint_string;
        return *this;
    }

    PropertyInfo& SetUsageFlags(const PropertyUsageFlags& property_usage_flags) {
        usage = property_usage_flags;
        return *this;
    }


    bool operator==(const PropertyInfo &property_info) const = default;

    bool operator<(const PropertyInfo &p_info) const {
        return name < p_info.name;
    }

};

static constexpr const char *PROPERTY_INFO_KEY  = "PROPERTY_INFO_KEY";

MetaData AddMetaPropertyInfo(const PropertyInfo& property_info);


class GOBOT_EXPORT Object : public QObject {
    GOBCLASS(Object)
public:

    Object();

    ~Object() override;

    [[nodiscard]] FORCE_INLINE std::string_view GetClassStringName() const { return get_type().get_name().data(); }

    [[nodiscard]] FORCE_INLINE Type GetType() const { return get_type(); }

    template <typename T, typename... Args>
    static T* New(Args&&... args) {
        static_assert(std::is_base_of_v<Object, T>, "Object must be the base of T");
        auto* obj = new T(std::forward<Args>(args)...);
        obj->PostNew();
        return obj;
    }

    template <typename T>
    static void Delete(T* object) {
        static_assert(std::is_base_of_v<Object, T>, "Object must be the base of T");
        object->PreDelete();
        delete object;
    }

    template <class T>
    static T* PointerCastTo(Object *object) {
        return dynamic_cast<T*>(object);
    }

    template <class T>
    static const T* PointerCastTo(const Object *object) {
        return dynamic_cast<const T*>(object);
    }

    bool Set(const String& name, Argument arg);

    static Type GetDerivedTypeByInstance(Instance instance) {
        auto raw_type = instance.get_type().get_raw_type();
        Instance obj = raw_type.is_wrapper() ? instance.get_wrapped_instance() : instance;
        return obj.get_derived_type();
    }

    static Instance GetWrapperInstance(Instance instance) {
        auto raw_type = instance.get_type().get_raw_type();
        return raw_type.is_wrapper() ? instance.get_wrapped_instance() : instance;
    }

    [[nodiscard]] Variant Get(const String& name) const;

    Type GetPropertyType(const String& name) const;

    [[nodiscard]] ALWAYS_INLINE bool IsRefCounted() const { return type_is_reference_; }

    [[nodiscard]] ALWAYS_INLINE ObjectID GetInstanceId() const { return instance_id_; }

protected:

    /// Notification
    void NotificationCallBack(NotificationType notification) {}

    static FORCE_INLINE auto GetNotificationCallBack() -> void(Object::*)(NotificationType)  {
        return &Object::NotificationCallBack;
    }

    virtual void NotificationImpl(NotificationType notification, bool reversed) {
    }

public:
    // Notification
    void Notification(NotificationType notification, bool reversed = false) {
        NotificationImpl(notification, reversed);
    }

private:
    void PostNew() {
        Notification(NotificationType::PostNew);
    }

    void PreDelete() {
        Notification(NotificationType::PreDelete, true);
    }


    FORCE_INLINE void ConstructObject(bool reference);

    explicit Object(bool reference);

private:
    friend class RefCounted;
    bool type_is_reference_ = false;
    ObjectID instance_id_;
};


class GOBOT_EXPORT ObjectDB {
// This needs to add up to 63, 1 bit is for reference.
#define OBJECTDB_VALIDATOR_BITS 39
#define OBJECTDB_VALIDATOR_MASK ((uint64_t(1) << OBJECTDB_VALIDATOR_BITS) - 1)
#define OBJECTDB_SLOT_MAX_COUNT_BITS 24
#define OBJECTDB_SLOT_MAX_COUNT_MASK ((uint64_t(1) << OBJECTDB_SLOT_MAX_COUNT_BITS) - 1)
#define OBJECTDB_REFERENCE_BIT (uint64_t(1) << (OBJECTDB_SLOT_MAX_COUNT_BITS + OBJECTDB_VALIDATOR_BITS))

public:

    struct ObjectSlot { // 128 bits per slot.
        uint64_t validator : OBJECTDB_VALIDATOR_BITS;
        uint64_t next_free : OBJECTDB_SLOT_MAX_COUNT_BITS;
        uint64_t is_ref_counted : 1;
        Object *object = nullptr;
    };


private:
    static ObjectID AddInstance(Object *object);

    static void RemoveInstance(Object *object);

private:
    friend class Object;

    static SpinLock s_spin_lock;
    static uint32_t s_slot_count;
    static uint32_t s_slot_max;
    static ObjectSlot* s_object_slots;
    static uint64_t s_validator_counter;

};

} // end of namespace gobot