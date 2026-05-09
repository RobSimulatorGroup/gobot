#include "test_render_backend.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

int main(int argc, char** argv) {
    gobot::test::RegisterTestRasterizer();
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}
