#include <gtest/gtest.h>

#include <gobot/core/notification_enum.hpp>
#include <gobot/core/object.hpp>

namespace gobot {

class TestObject : public Object {
    GOBCLASS(TestObject, Object)
};

} // namespace gobot

TEST(TestObjectDB, ResolvesLiveObjectByGenerationalId) {
    auto* object = gobot::Object::New<gobot::TestObject>();
    const gobot::ObjectID id = object->GetInstanceId();

    EXPECT_EQ(gobot::ObjectDB::GetInstance(id), object);

    gobot::Object::Delete(object);
}

TEST(TestObjectDB, DeletedObjectIdDoesNotResolve) {
    auto* object = gobot::Object::New<gobot::TestObject>();
    const gobot::ObjectID id = object->GetInstanceId();

    gobot::Object::Delete(object);

    EXPECT_EQ(gobot::ObjectDB::GetInstance(id), nullptr);
}

TEST(TestObjectDB, ReusedSlotDoesNotResolveOldId) {
    auto* first = gobot::Object::New<gobot::TestObject>();
    const gobot::ObjectID first_id = first->GetInstanceId();
    gobot::Object::Delete(first);

    auto* second = gobot::Object::New<gobot::TestObject>();
    const gobot::ObjectID second_id = second->GetInstanceId();

    EXPECT_EQ(gobot::ObjectDB::GetInstance(first_id), nullptr);
    EXPECT_EQ(gobot::ObjectDB::GetInstance(second_id), second);
    EXPECT_NE(first_id, second_id);

    gobot::Object::Delete(second);
}
