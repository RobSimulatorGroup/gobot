/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-10
*/

#pragma once

#include "imgui.h"

namespace gobot {



}

static inline ImVec2 operator*(const ImVec2& lhs, const float rhs)
{
    return ImVec2(lhs.x * rhs, lhs.y * rhs);
}
static inline ImVec2 operator/(const ImVec2& lhs, const float rhs)
{
    return ImVec2(lhs.x / rhs, lhs.y / rhs);
}
static inline ImVec2 operator+(const ImVec2& lhs, const ImVec2& rhs)
{
    return ImVec2(lhs.x + rhs.x, lhs.y + rhs.y);
}
static inline ImVec2 operator-(const ImVec2& lhs, const ImVec2& rhs)
{
    return ImVec2(lhs.x - rhs.x, lhs.y - rhs.y);
}
static inline ImVec2 operator*(const ImVec2& lhs, const ImVec2& rhs)
{
    return ImVec2(lhs.x * rhs.x, lhs.y * rhs.y);
}
static inline ImVec2 operator/(const ImVec2& lhs, const ImVec2& rhs)
{
    return ImVec2(lhs.x / rhs.x, lhs.y / rhs.y);
}
static inline ImVec2& operator+=(ImVec2& lhs, const ImVec2& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    return lhs;
}
static inline ImVec2& operator-=(ImVec2& lhs, const ImVec2& rhs)
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    return lhs;
}
static inline ImVec2& operator*=(ImVec2& lhs, const float rhs)
{
    lhs.x *= rhs;
    lhs.y *= rhs;
    return lhs;
}
static inline ImVec2& operator/=(ImVec2& lhs, const float rhs)
{
    lhs.x /= rhs;
    lhs.y /= rhs;
    return lhs;
}
static inline ImVec4 operator-(const ImVec4& lhs, const ImVec4& rhs)
{
    return ImVec4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
}
static inline ImVec4 operator+(const ImVec4& lhs, const ImVec4& rhs)
{
    return ImVec4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}
static inline ImVec4 operator*(const ImVec4& lhs, const float rhs)
{
    return ImVec4(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs);
}
static inline ImVec4 operator/(const ImVec4& lhs, const float rhs)
{
    return ImVec4(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs);
}
static inline ImVec4 operator*(const ImVec4& lhs, const ImVec4& rhs)
{
    return ImVec4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);
}
static inline ImVec4 operator/(const ImVec4& lhs, const ImVec4& rhs)
{
    return ImVec4(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w);
}
static inline ImVec4& operator+=(ImVec4& lhs, const ImVec4& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    lhs.w += rhs.w;
    return lhs;
}
static inline ImVec4& operator-=(ImVec4& lhs, const ImVec4& rhs)
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    lhs.w -= rhs.w;
    return lhs;
}
static inline ImVec4& operator*=(ImVec4& lhs, const float rhs)
{
    lhs.x *= rhs;
    lhs.y *= rhs;
    return lhs;
}
static inline ImVec4& operator/=(ImVec4& lhs, const float rhs)
{
    lhs.x /= rhs;
    lhs.y /= rhs;
    return lhs;
}

