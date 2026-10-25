#include <gtest/gtest.h>
#include "gobot/core/io/image.hpp"

TEST(Image, NonPowerOfTwoRgbMipChainTerminatesAndHasExactSize) {
    using namespace gobot;
    EXPECT_EQ(Image::GetImageDataSize(320, 240, ImageFormat::RGB8, false), 230400);
    EXPECT_EQ(Image::GetImageDataSize(320, 240, ImageFormat::RGB8, true), 307164);
    EXPECT_EQ(Image::GetImageRequiredMipmaps(320, 240, ImageFormat::RGB8), 8);
    Image image(320, 240, true, ImageFormat::RGB8);
    EXPECT_EQ(image.GetMipmapCount(), 8);
    EXPECT_EQ(image.GetMipmapByteSize(8), 3);
    EXPECT_EQ(image.GetMipmapOffset(8), 307161);
}

TEST(Image, SinglePixelAndThinMipChainsTerminate) {
    using namespace gobot;
    EXPECT_EQ(Image::GetImageDataSize(1, 1, ImageFormat::RGB8, true), 3);
    EXPECT_EQ(Image::GetImageDataSize(1, 7, ImageFormat::RGB8, true), 33);
    EXPECT_EQ(Image::GetImageRequiredMipmaps(1, 7, ImageFormat::RGB8), 2);
}
