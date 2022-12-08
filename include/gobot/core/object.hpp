/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-9-24
*/

#pragma once

#include "gobot/core/types.hpp"
#include "gobot/core/marcos.hpp"
#include "notification_enum.hpp"
#include <rttr/rttr_enable.h>

#include <QObject>

// Do nothing for 1
#define GOBOT_CLASS_1(...)    \


#define GOBOT_CLASS_2(Class, BaseClass)                                                             \
protected:                                                                                          \
    static FORCE_INLINE auto GetNotificationCallback() -> void(Object::*)(int) {                    \
         return (void(Object::*)(int)) &Class::NotificationCallBack;                                \
    }                                                                                               \
    void NotificationImpl(NotificationType notification, bool reversed) override {                  \
        if (!reversed) { BaseClass::NotificationImpl(notification, reversed); }                     \
                                                                                                    \
	    if (Class::GetNotificationCallback() != BaseClass::GetNotificationCallback()) {             \
            Notification(notification);                                                             \
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


enum class GOBOT_EXPORT PropertyHint {
    None, ///< no hint provided.
    Range, ///< hint_text = "min,max[,step][,or_greater][,or_less][,hide_slider][,radians][,degrees][,exp][,suffix:<keyword>] range.
    Flags, ///< hint_text= "flag1,flag2,etc" (as bit flags)
    File, ///< a file path must be passed, hint_text (optionally) is a filter "*.png,*.wav,*.doc,"
    Dir, ///< a directory path must be passed
    GlobalFile, ///< a file path must be passed, hint_text (optionally) is a filter "*.png,*.wav,*.doc,"
    GlobalDir, ///< a directory path must be passed
};

enum class GOBOT_EXPORT PropertyUsageFlags {
    None = 0,
    Storage = 1 << 1,
    Editor = 1 << 2,
    UsageDefault = Storage | Editor,
};


struct GOBOT_EXPORT PropertyInfo {
    String name;
    PropertyHint hint = PropertyHint::None;
    String hint_string;
    PropertyUsageFlags usage = PropertyUsageFlags::None;

    PropertyInfo& SetName(const String& _name) {
        name = _name;
        return *this;
    }

    PropertyInfo& SetHint(const PropertyHint& property_hint) {
        hint = property_hint;
        return *this;
    }

    PropertyInfo& SetHint(const String& _hint_string) {
        hint_string = _hint_string;
        return *this;
    }

    PropertyInfo& SetHint(const PropertyUsageFlags& property_usage_flags) {
        usage = property_usage_flags;
        return *this;
    }


    bool operator==(const PropertyInfo &property_info) const {
        return  (name == property_info.name) &&
                (hint == property_info.hint) &&
                (hint_string == property_info.hint_string) &&
                (usage == property_info.usage);
    }

    bool operator<(const PropertyInfo &p_info) const {
        return name < p_info.name;
    }

};


class GOBOT_EXPORT Object : public QObject {
    GOBCLASS(Object)
public:

    Object();

    [[nodiscard]] FORCE_INLINE std::string_view GetClassName() const { return get_type().get_name().data(); }

    [[nodiscard]] FORCE_INLINE Type GetType() const { return get_type(); }

    template <typename T, typename... Args>
    static std::enable_if_t<std::is_base_of_v<Object, T>, T*> New(Args&&... args) {
        auto* obj = new T(std::forward<Args>(args)...);
        return obj->PostNew();
    }

    template <typename T, typename = std::enable_if_t<std::is_base_of_v<Object, T>>>
    static void Delete(T* object) {
        object->PreDelete();
        delete object;
    }

    template <class T>
    static T *CastTo(Object *object) {
        return dynamic_cast<T *>(object);
    }

    template <class T>
    static const T *CastTo(const Object *object) {
        return dynamic_cast<const T *>(object);
    }

    bool SetProperty(const StringName& name, Argument argument);

    bool GetProperty(const StringName& name, Variant& varint);

protected:

    /// Notification
    void NotificationCallBack(int notification) {}

    static FORCE_INLINE auto GetNotificationCallback() -> void(Object::*)(int)  {
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

};


} // end of namespace gobot::core