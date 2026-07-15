#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include <gobot/core/math/pcg_random.hpp>

TEST(Pcg64Random, matches_numpy_pcg64_raw_sequence) {
    gobot::Pcg64Random random(42);
    constexpr std::array<std::uint64_t, 8> expected{
            14276969152011380360ULL,
            8095878257575067585ULL,
            15838336090824644132ULL,
            12864169557245331597ULL,
            1737265434024182251ULL,
            17997055833233904524ULL,
            14040549286955598961ULL,
            14500327064922265408ULL,
    };
    for (std::uint64_t value : expected) {
        EXPECT_EQ(random.NextUInt64(), value);
    }
}

TEST(Pcg64Random, matches_numpy_default_rng_uniform_sequence) {
    gobot::Pcg64Random random(42);
    constexpr std::array<double, 8> expected{
            0.7739560485559633,
            0.4388784397520523,
            0.8585979199113825,
            0.6973680290593639,
            0.09417734788764953,
            0.9756223516367559,
            0.761139701990353,
            0.7860643052769538,
    };
    for (double value : expected) {
        EXPECT_DOUBLE_EQ(random.Uniform(), value);
    }
}

TEST(Pcg64Random, matches_numpy_bounded_integer_sequence) {
    gobot::Pcg64Random random(42);
    constexpr std::array<std::uint32_t, 40> expected{
            0, 3, 3, 2, 2, 4, 0, 3, 1, 0,
            2, 4, 3, 3, 3, 3, 2, 0, 4, 2,
            2, 1, 0, 4, 3, 3, 2, 4, 2, 2,
            2, 1, 0, 2, 4, 0, 4, 4, 1, 3,
    };
    for (std::uint32_t value : expected) {
        EXPECT_EQ(random.BoundedUInt32(5), value);
    }
}
