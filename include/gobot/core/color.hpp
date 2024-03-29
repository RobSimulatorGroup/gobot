/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-23
*/

#pragma once

#include "gobot/core/macros.hpp"
#include "gobot_export.h"

namespace gobot {

class GOBOT_EXPORT Color {
public:

    Color() = default;

    Color(const Color& color) = default;

    Color& operator=(const Color& color) = default;

    Color(float r, float g, float b, float a = 1.0)
        : r_(r),
          g_(g),
          b_(b),
          a_(a) {
    }

    /**
     * Creates the Color ``(r, g, b, a) / 255.0``.
     * Values are casted to floats before division.
     */
    Color(int r, int g, int b, int a = 255)
        : r_(r / 255.f),
          g_(g / 255.f),
          b_(b / 255.f),
          a_(a / 255.f) {
    }

    /**
     * Creates the Color ``(intensity, intensity, intensity, alpha)``.
     */
    Color(float intensity, float alpha)
            : Color(intensity, intensity, intensity, alpha) {
    }

    /**
     * Creates the Color ``(intensity, intensity, intensity, alpha) / 255.0``.
     * Values are casted to floats before division.
     */
    Color(int intensity, int alpha)
        : Color(intensity, intensity, intensity, alpha) {
    }

    static Color FromRGBE9995(uint32_t rgbe);

    uint32_t ToRGBE9995() const;

    /// Return a reference to the red channel
    float& red() { return r_; }

    /// Return a reference to the red channel (const version)
    [[nodiscard]] const float& red() const { return r_; }

    /// Return a reference to the green channel
    float& green() { return g_; }

    /// Return a reference to the green channel (const version)
    [[nodiscard]] const float& green() const { return g_; }

    /// Return a reference to the blue channel
    float& blue() { return b_; }

    /// Return a reference to the blue channel (const version)
    [[nodiscard]] const float& blue() const { return b_; }

    /// Return a reference to the alpha channel
    float& alpha() { return a_; }

    /// Return a reference to the alpha channel (const version)
    [[nodiscard]] const float& alpha() const { return a_; }

    [[nodiscard]] FORCE_INLINE  std::uint32_t GetPackedRgbA() const {
        return (std::uint8_t(r_ * 255) << 24) + (std::uint8_t(g_ * 255) << 16) + (std::uint8_t(b_ * 255) << 8) + (std::uint8_t(a_ * 255));
    }

    float hue() const;

    float saturation() const;

    float val() const;

    Color operator*(const Color &color) const {
        return {r_ * color.red(),
                g_ * color.green(),
                b_ * color.blue(),
                a_ * color.alpha()};
    }

    Color operator*(float scalar) const {
        return {r_ * scalar,
                g_ * scalar,
                b_ * scalar,
                a_ * scalar};
    }

    void operator*=(const Color &color) {
        r_ = r_ * color.red();
        g_ = g_ * color.green();
        b_ = b_ * color.blue();
        a_ = a_ * color.alpha();
    }

    void operator*=(float scalar) {
        r_ = r_ * scalar;
        g_ = g_ * scalar;
        b_ = b_ * scalar;
        a_ = a_ * scalar;
    }

    Color operator/(const Color &color) const {
        return {r_ / color.red(),
                g_ / color.green(),
                b_ / color.blue(),
                a_ / color.alpha()};
    }

    Color operator/(float scalar) const {
        return {r_ / scalar,
                g_ / scalar,
                b_ / scalar,
                a_ / scalar};
    }

    void operator/=(const Color &color) {
        r_ = r_ / color.red();
        g_ = g_ / color.green();
        b_ = b_ / color.blue();
        a_ = a_ / color.alpha();
    }

    void operator/=(float scalar) {
        r_ = r_ / scalar;
        g_ = g_ / scalar;
        b_ = b_ / scalar;
        a_ = a_ / scalar;
    }

    Color operator+(const Color &color) const {
        return {r_ + color.red(),
                g_ + color.green(),
                b_ + color.blue(),
                a_ + color.alpha()};
    }

    void operator+=(const Color &color) {
        r_ = r_ + color.red();
        g_ = g_ + color.green();
        b_ = b_ + color.blue();
        a_ = a_ + color.alpha();
    }

    Color operator-() const {
        return {1.0f - r_,
                1.0f - g_,
                1.0f - b_,
                1.0f - a_};
    }

    Color operator-(const Color &color) const {
        return {r_ - color.red(),
                g_ - color.green(),
                b_ - color.blue(),
                a_ - color.alpha()};
    }

    void operator-=(const Color &color) {
        r_ = r_ - color.red();
        g_ = g_ - color.green();
        b_ = b_ - color.blue();
        a_ = a_ - color.alpha();
    }

private:
    GOBOT_REGISTRATION_FRIEND

    float r_{0.0};
    float g_{0.0};
    float b_{0.0};
    float a_{0.0};
};

}