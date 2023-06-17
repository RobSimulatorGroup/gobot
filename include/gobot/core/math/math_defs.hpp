/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Yingnan
 * Wu<wuyingnan@users.noreply.github.com>. Everyone is permitted to copy and
 * distribute verbatim copies of this license document, but changing it is not
 * allowed. This version of the GNU Lesser General Public License incorporates
 * the terms and conditions of version 3 of the GNU General Public License. This
 * file is created by Yingnan Wu, 23-2-12
 */

#pragma once

#include "Eigen/Eigen"

namespace gobot {

#define CMP_EPSILON 0.00001
#define CMP_EPSILON2 (CMP_EPSILON * CMP_EPSILON)

#define CMP_NORMALIZE_TOLERANCE 0.000001
#define CMP_POINT_IN_PLANE_EPSILON 0.00001

#define Math_SQRT12 0.7071067811865475244008443621048490
#define Math_SQRT2 1.4142135623730950488016887242
#define Math_LN2 0.6931471805599453094172321215
#define Math_TAU 6.2831853071795864769252867666
#define Math_PI 3.1415926535897932384626433833
/// Pi divided by two. pi/2
#define Math_HALF_PI 1.5707963267948966192313216916398f
#define Math_E 2.7182818284590452353602874714

#ifdef DEBUG_ENABLED
#define MATH_CHECKS
#endif

// this epsilon is for values related to a unit size (scalar or vector len)
#ifdef PRECISE_MATH_CHECKS
#define UNIT_EPSILON 0.00001
#else
// tolerate some more floating point error normally
#define UNIT_EPSILON 0.001
#endif

#define USEC_TO_SEC(m_usec) ((m_usec) / 1000000.0)

enum class ClockDirection { Clockwise, CounterClockwise };

enum class Orientation { Horizontal, Vertical };

enum class HorizontalAlignment {
  Left,
  Center,
  Right,
  Fill,
};

enum class VerticalAlignment {
  Top,
  Center,
  Bottom,
  Fill,
};

enum class InlineAlignment {
  // Image alignment points.
  TopTo = 0b0000,
  CenterTo = 0b0001,
  BaselineTo = 0b0011,
  BottomTo = 0b0010,
  ImageMask = 0b0011,

  // Text alignment points.
  ToTop = 0b0000,
  ToCenter = 0b0100,
  ToBaseline = 0b1000,
  ToBottom = 0b1100,
  TextMask = 0b1100,

  // Presets.
  Top = TopTo | ToTop,
  Center = CenterTo | ToCenter,
  Bottom = BottomTo | ToBottom
};

enum class Side { Left, Top, Right, Bottom };

enum class Corner { TopLeft, TopRight, BottomRight, BottomLeft };

// We follow Tait–Bryan angles
// https://en.wikipedia.org/wiki/Euler_angles
enum class EulerOrder {
    // rotated axis, (intrinsic rotations)
    RXYZ,
    RYZX,
    RZYX,
    RXZY,
    RXZX,
    RYXZ,

    // static axis (extrinsic rotations)
    SXYZ,
    SYZX,
    SZYX,
    SXZY,
    SXZX,
    SYXZ
};

enum class Handedness
{
    Left,
    Right,
};

enum class Axis
{
    X,
    Y,
    Z,

    Count
};


/**
 * The "Real" type is an abstract type used for real numbers, such as 1.5,
 * in contrast to integer numbers. Precision can be controlled with the
 * presence or absence of the REAL_T_IS_DOUBLE define.
 */
#ifdef REAL_T_IS_DOUBLE
using real_t = double;
#else
using real_t = float;
#endif

template <typename _Scalar = real_t>
struct MatrixData {
    Eigen::Index rows;
    Eigen::Index cols;
    std::vector<_Scalar> storage;
};


template <typename T>
constexpr int Sign(const T val) {
    return (static_cast<T>(0) < val) - (val < static_cast<T>(0));
}

inline real_t DEG_TO_RAD(real_t deg) { return static_cast<real_t>(deg * (Math_PI / 180.0)); }
inline real_t RAD_TO_DEG(real_t rad) { return static_cast<real_t>(rad * (180.0 / Math_PI)); }

template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
AlmostEqual(T x, T y, int ulp = 2)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::fabs(x - y) <= std::numeric_limits<T>::epsilon() * std::fabs(x + y) * ulp
           // unless the result is subnormal
           || std::fabs(x - y) < std::numeric_limits<T>::min();
}

};  // namespace gobot
