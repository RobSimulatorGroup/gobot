/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-23
*/

#include "gobot/core/color.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/core/math/math_defs.hpp"
#include "gobot/core/math/math_util.hpp"
#include <cmath>

namespace gobot {

Color Color::FromRGBE9995(uint32_t rgbe) {
    float r = rgbe & 0x1ff;
    float g = (rgbe >> 9) & 0x1ff;
    float b = (rgbe >> 18) & 0x1ff;
    float e = (rgbe >> 27);
    float m = std::pow(2.0f, e - 15.0f - 9.0f);

    float rd = r * m;
    float gd = g * m;
    float bd = b * m;

    return {rd, gd, bd, 1.0f};
}

uint32_t Color::ToRGBE9995() const {
    const float pow2to9 = 512.0f;
    const float B = 15.0f;
    const float N = 9.0f;

    float sharedexp = 65408.000f; // Result of: ((pow2to9 - 1.0f) / pow2to9) * powf(2.0f, 31.0f - 15.0f)

    float cRed = std::max(0.0f, std::min(sharedexp, r_));
    float cGreen = std::max(0.0f, std::min(sharedexp, g_));
    float cBlue = std::max(0.0f, std::min(sharedexp, b_));

    float cMax = std::max(cRed, std::max(cGreen, cBlue));

    float expp = std::max(-B - 1.0f, std::floor(std::log(cMax) / (real_t)Math_LN2)) + 1.0f + B;

    float sMax = (float)floor((cMax / std::pow(2.0f, expp - B - N)) + 0.5f);

    float exps = expp + 1.0f;

    if (0.0f <= sMax && sMax < pow2to9) {
        exps = expp;
    }

    float sRed = std::floor((cRed / pow(2.0f, exps - B - N)) + 0.5f);
    float sGreen = std::floor((cGreen / pow(2.0f, exps - B - N)) + 0.5f);
    float sBlue = std::floor((cBlue / pow(2.0f, exps - B - N)) + 0.5f);

    return (uint32_t(fast_ftoi(sRed)) & 0x1FF) |
           ((uint32_t(fast_ftoi(sGreen)) & 0x1FF) << 9) |
           ((uint32_t(fast_ftoi(sBlue)) & 0x1FF) << 18) |
           ((uint32_t(fast_ftoi(exps)) & 0x1F) << 27);
}

float Color::hue() const {
    float min = std::min(r_, g_);
    min = std::min(min, b_);
    float max = std::max(r_, g_);
    max = std::max(max, b_);

    float delta = max - min;

    if (delta == 0.0f) {
        return 0.0f;
    }

    float h;
    if (r_ == max) {
        h = (g_ - b_) / delta; // between yellow & magenta
    } else if (g_ == max) {
        h = 2 + (b_ - r_) / delta; // between cyan & yellow
    } else {
        h = 4 + (r_ - g_) / delta; // between magenta & cyan
    }

    h /= 6.0f;
    if (h < 0.0f) {
        h += 1.0f;
    }

    return h;
}

float Color::saturation() const {
    float min = std::min(r_, g_);
    min = std::min(min, b_);
    float max = std::max(r_, g_);
    max = std::max(max, b_);

    float delta = max - min;

    return (max != 0.0f) ? (delta / max) : 0.0f;
}

float Color::val() const {
    float max = std::max(r_, g_);
    max = std::max(max, b_);
    return max;
}

}


GOBOT_REGISTRATION {
    Class_<Color>("Color")
        .constructor()(CtorAsObject)
        .property("red", &Color::r_)
        .property("green", &Color::g_)
        .property("blue", &Color::b_)
        .property("alpha", &Color::a_);

};
