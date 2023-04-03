/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-20
*/

#include "gobot/core/object.hpp"

#include <utility>

#include "gobot/core/notification_enum.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/log.hpp"

namespace gobot {

MetaData AddMetaPropertyInfo(const PropertyInfo& property_info) {
    return rttr::metadata(PROPERTY_INFO_KEY, property_info);
}

Object::Object() {
    ConstructObject(false);
}

void Object::ConstructObject(bool reference) {
    type_is_reference_ = reference;
    instance_id_ = ObjectDB::AddInstance(this);
}

Object::Object(bool reference) {
    ConstructObject(reference);
}

Object::~Object() {
    if (instance_id_ != ObjectID()) {
        ObjectDB::RemoveInstance(this);
        instance_id_ = ObjectID();
    }
}

bool Object::Set(const String& name, Argument arg) {
    auto type = GetType();
    return type.set_property_value(name.toStdString(), Instance(this), std::move(arg));
}


Variant Object::Get(const String& name) const {
    Variant res;

    auto type = GetType();
    res = type.get_property_value(name.toStdString(), Instance(this));

    return res;
}

Type Object::GetPropertyType(const String& name) const {
    auto property = GetType().get_property(name.toStdString());
    return property.get_type();
}


SpinLock ObjectDB::s_spin_lock;
uint32_t ObjectDB::s_slot_count;
uint32_t ObjectDB::s_slot_max;
ObjectDB::ObjectSlot* ObjectDB::s_object_slots = nullptr;
uint64_t ObjectDB::s_validator_counter = 0;

ObjectID ObjectDB::AddInstance(Object *object) {
    s_spin_lock.lock();

    // realloc to 2 * slots
    if (s_slot_count == s_slot_max) [[unlikely]] {
        if (s_slot_count == (1 << OBJECTDB_SLOT_MAX_COUNT_BITS)) {
            LOG_FATAL("slot_count to max slot count");
            GENERATE_TRAP();
        }

        uint32_t new_slot_max = s_slot_max > 0 ? s_slot_max * 2 : 1;

        s_object_slots = (ObjectSlot *)realloc(s_object_slots, sizeof(ObjectSlot) * new_slot_max);
        for (uint32_t i = s_slot_max; i < new_slot_max; i++) {
            s_object_slots[i].object = nullptr;
            s_object_slots[i].is_ref_counted = false;
            s_object_slots[i].next_free = i;
            s_object_slots[i].validator = 0;
        }
        s_slot_max = new_slot_max;
    }

    uint32_t slot = s_object_slots[s_slot_count].next_free;
    if (s_object_slots[slot].object != nullptr) {
        s_spin_lock.unlock();
        if (s_object_slots[slot].object != nullptr) [[unlikely]] {
            LOG_ERROR("The slot placed has a not null object");
            return {};
        }
    }
    s_object_slots[slot].object = object;
    s_object_slots[slot].is_ref_counted = object->IsRefCounted();
    s_validator_counter = (s_validator_counter + 1) & OBJECTDB_VALIDATOR_MASK;
    if (s_validator_counter == 0) [[unlikely]] {
        s_validator_counter = 1;
    }
    s_object_slots[slot].validator = s_validator_counter;

    uint64_t id = s_validator_counter;
    id <<= OBJECTDB_SLOT_MAX_COUNT_BITS;
    id |= uint64_t(slot);

    if (object->IsRefCounted()) {
        id |= OBJECTDB_REFERENCE_BIT;
    }

    s_slot_count++;

    s_spin_lock.unlock();
    return ObjectID(id);
}

void ObjectDB::RemoveInstance(Object *object) {
    uint64_t t = object->GetInstanceId();
    uint32_t slot = t & OBJECTDB_SLOT_MAX_COUNT_MASK; //slot is always valid on valid object

    s_spin_lock.lock();

    //decrease slot count
    s_slot_count--;
    //set the free slot properly
    s_object_slots[s_slot_count].next_free = slot;
    //invalidate, so checks against it fail
    s_object_slots[slot].validator = 0;
    s_object_slots[slot].is_ref_counted = false;
    s_object_slots[slot].object = nullptr;

    s_spin_lock.unlock();
}


}

GOBOT_REGISTRATION {

    QuickEnumeration_<PropertyHint>("PropertyHint");

    QuickEnumeration_<PropertyUsageFlags>("PropertyUsageFlags");

    Class_<PropertyInfo>("PropertyInfo")
            .constructor()(CtorAsObject)
            .property("name", &gobot::PropertyInfo::name)
            .property("hint", &gobot::PropertyInfo::hint)
            .property("hint_string", &gobot::PropertyInfo::hint_string)
            .property("PropertyUsageFlags", &gobot::PropertyInfo::usage);

    Class_<Object>("Object")
            .constructor()(CtorAsRawPtr);


    QuickEnumeration_<NotificationType>("NotificationType");

};
