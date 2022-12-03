/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-6
*/

#pragma once

#include <rttr/rttr_enable.h>

// Should always inline no matter what.
#ifndef ALWAYS_INLINE
#if defined(__GNUC__)
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define ALWAYS_INLINE __forceinline
#else
#define ALWAYS_INLINE inline
#endif
#endif

// Should always inline, except in dev builds because it makes debugging harder.
#ifndef FORCE_INLINE
#ifdef NDEBUG
#define FORCE_INLINE ALWAYS_INLINE
#else
#define FORCE_INLINE inline
#endif
#endif


// C++ preprocessor __VA_ARGS__ number of arguments
#define PP_NARG_COUNT(...) PP_NARG_(__VA_ARGS__,PP_RSEQ_N())
#define PP_NARG_(...) PP_ARG_N(__VA_ARGS__)
#define PP_ARG_N( _1, _2, _3, _4, _5, _6, _7, _8, _9,_10, _11,_12,_13,_14,_15,_16,_17,_18,_19,_20, N,...) N
#define PP_RSEQ_N() 20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0


//  concatenation
#define CAT2(A, B) A##B
#define CAT3(A, B, C) A##B##C
#define CAT4(A, B, C, D) A##B##C##D
#define CAT5(A, B, C, D, E) A##B##C##D##E

#define GOB_UNUSED(x) (void)x;