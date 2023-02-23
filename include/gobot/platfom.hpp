/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-23
*/

#pragma once

// Basically copy from bgfx/platform.h

// Architecture
#define GOB_ARCH_32BIT 0
#define GOB_ARCH_64BIT 0

// Compiler
#define GOB_COMPILER_CLANG          0
#define GOB_COMPILER_CLANG_ANALYZER 0
#define GOB_COMPILER_GCC            0
#define GOB_COMPILER_MSVC           0

// Endianness
#define GOB_CPU_ENDIAN_BIG    0
#define GOB_CPU_ENDIAN_LITTLE 0

// CPU
#define GOB_CPU_ARM   0
#define GOB_CPU_JIT   0
#define GOB_CPU_MIPS  0
#define GOB_CPU_PPC   0
#define GOB_CPU_RISCV 0
#define GOB_CPU_X86   0

// C Runtime
#define GOB_CRT_BIONIC 0
#define GOB_CRT_BSD    0
#define GOB_CRT_GLIBC  0
#define GOB_CRT_LIBCXX 0
#define GOB_CRT_MINGW  0
#define GOB_CRT_MSVC   0
#define GOB_CRT_NEWLIB 0

#ifndef GOB_CRT_NONE
#	define GOB_CRT_NONE 0
#endif // GOB_CRT_NONE

// Language standard version
#define GOB_LANGUAGE_CPP14 201402L
#define GOB_LANGUAGE_CPP17 201703L
#define GOB_LANGUAGE_CPP20 202002L
#define GOB_LANGUAGE_CPP23 202207L

// Platform
#define GOB_PLATFORM_ANDROID    0
#define GOB_PLATFORM_BSD        0
#define GOB_PLATFORM_EMSCRIPTEN 0
#define GOB_PLATFORM_HAIKU      0
#define GOB_PLATFORM_HURD       0
#define GOB_PLATFORM_IOS        0
#define GOB_PLATFORM_LINUX      0
#define GOB_PLATFORM_NX         0
#define GOB_PLATFORM_OSX        0
#define GOB_PLATFORM_PS4        0
#define GOB_PLATFORM_PS5        0
#define GOB_PLATFORM_RPI        0
#define GOB_PLATFORM_WINDOWS    0
#define GOB_PLATFORM_WINRT      0
#define GOB_PLATFORM_XBOXONE    0

// http://sourceforge.net/apps/mediawiki/predef/index.php?title=Compilers
#if defined(__clang__)
// clang defines __GNUC__ or _MSC_VER
#	undef  GOB_COMPILER_CLANG
#	define GOB_COMPILER_CLANG (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#	if defined(__clang_analyzer__)
#		undef  GOB_COMPILER_CLANG_ANALYZER
#		define GOB_COMPILER_CLANG_ANALYZER 1
#	endif // defined(__clang_analyzer__)
#elif defined(_MSC_VER)
#	undef  GOB_COMPILER_MSVC
#	define GOB_COMPILER_MSVC _MSC_VER
#elif defined(__GNUC__)
#	undef  GOB_COMPILER_GCC
#	define GOB_COMPILER_GCC (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#else
#	error "GOB_COMPILER_* is not defined!"
#endif //

// http://sourceforge.net/apps/mediawiki/predef/index.php?title=Architectures
#if defined(__arm__)     \
 || defined(__aarch64__) \
 || defined(_M_ARM)
#	undef  GOB_CPU_ARM
#	define GOB_CPU_ARM 1
#	define GOB_CACHE_LINE_SIZE 64
#elif defined(__MIPSEL__)     \
 ||   defined(__mips_isa_rev) \
 ||   defined(__mips64)
#	undef  GOB_CPU_MIPS
#	define GOB_CPU_MIPS 1
#	define GOB_CACHE_LINE_SIZE 64
#elif defined(_M_PPC)        \
 ||   defined(__powerpc__)   \
 ||   defined(__powerpc64__)
#	undef  GOB_CPU_PPC
#	define GOB_CPU_PPC 1
#	define GOB_CACHE_LINE_SIZE 128
#elif defined(__riscv)   \
 ||   defined(__riscv__) \
 ||   defined(RISCVEL)
#	undef  GOB_CPU_RISCV
#	define GOB_CPU_RISCV 1
#	define GOB_CACHE_LINE_SIZE 64
#elif defined(_M_IX86)    \
 ||   defined(_M_X64)     \
 ||   defined(__i386__)   \
 ||   defined(__x86_64__)
#	undef  GOB_CPU_X86
#	define GOB_CPU_X86 1
#	define GOB_CACHE_LINE_SIZE 64
#else // PNaCl doesn't have CPU defined.
#	undef  GOB_CPU_JIT
#	define GOB_CPU_JIT 1
#	define GOB_CACHE_LINE_SIZE 64
#endif //

#if defined(__x86_64__)    \
 || defined(_M_X64)        \
 || defined(__aarch64__)   \
 || defined(__64BIT__)     \
 || defined(__mips64)      \
 || defined(__powerpc64__) \
 || defined(__ppc64__)     \
 || defined(__LP64__)
#	undef  GOB_ARCH_64BIT
#	define GOB_ARCH_64BIT 64
#else
#	undef  GOB_ARCH_32BIT
#	define GOB_ARCH_32BIT 32
#endif //

#if GOB_CPU_PPC
// __BIG_ENDIAN__ is gcc predefined macro
#	if defined(__BIG_ENDIAN__)
#		undef  GOB_CPU_ENDIAN_BIG
#		define GOB_CPU_ENDIAN_BIG 1
#	else
#		undef  GOB_CPU_ENDIAN_LITTLE
#		define GOB_CPU_ENDIAN_LITTLE 1
#	endif
#else
#	undef  GOB_CPU_ENDIAN_LITTLE
#	define GOB_CPU_ENDIAN_LITTLE 1
#endif // GOB_CPU_PPC

// http://sourceforge.net/apps/mediawiki/predef/index.php?title=Operating_Systems
#if defined(_DURANGO) || defined(_XBOX_ONE)
#	undef  GOB_PLATFORM_XBOXONE
#	define GOB_PLATFORM_XBOXONE 1
#elif defined(_WIN32) || defined(_WIN64)
// http://msdn.microsoft.com/en-us/library/6sehtctf.aspx
#	ifndef NOMINMAX
#		define NOMINMAX
#	endif // NOMINMAX
//  If _USING_V110_SDK71_ is defined it means we are using the v110_xp or v120_xp toolset.
#	if defined(_MSC_VER) && (_MSC_VER >= 1700) && (!_USING_V110_SDK71_)
#		include <winapifamily.h>
#	endif // defined(_MSC_VER) && (_MSC_VER >= 1700) && (!_USING_V110_SDK71_)
#	if !defined(WINAPI_FAMILY) || (WINAPI_FAMILY == WINAPI_FAMILY_DESKTOP_APP)
#		undef  GOB_PLATFORM_WINDOWS
#		if !defined(WINVER) && !defined(_WIN32_WINNT)
#			if GOB_ARCH_64BIT
//				When building 64-bit target Win7 and above.
#				define WINVER 0x0601
#				define _WIN32_WINNT 0x0601
#			else
//				Windows Server 2003 with SP1, Windows XP with SP2 and above
#				define WINVER 0x0502
#				define _WIN32_WINNT 0x0502
#			endif // GOB_ARCH_64BIT
#		endif // !defined(WINVER) && !defined(_WIN32_WINNT)
#		define GOB_PLATFORM_WINDOWS _WIN32_WINNT
#	else
#		undef  GOB_PLATFORM_WINRT
#		define GOB_PLATFORM_WINRT 1
#	endif
#elif defined(__ANDROID__)
// Android compiler defines __linux__
#	include <sys/cdefs.h> // Defines __BIONIC__ and includes android/api-level.h
#	undef  GOB_PLATFORM_ANDROID
#	define GOB_PLATFORM_ANDROID __ANDROID_API__
#elif defined(__VCCOREVER__)
// RaspberryPi compiler defines __linux__
#	undef  GOB_PLATFORM_RPI
#	define GOB_PLATFORM_RPI 1
#elif  defined(__linux__)
#	undef  GOB_PLATFORM_LINUX
#	define GOB_PLATFORM_LINUX 1
#elif  defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__) \
	|| defined(__ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__)
#	undef  GOB_PLATFORM_IOS
#	define GOB_PLATFORM_IOS 1
#elif defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__)
#	undef  GOB_PLATFORM_OSX
#	define GOB_PLATFORM_OSX __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__
#elif defined(__EMSCRIPTEN__)
#	undef  GOB_PLATFORM_EMSCRIPTEN
#	define GOB_PLATFORM_EMSCRIPTEN 1
#elif defined(__ORBIS__)
#	undef  GOB_PLATFORM_PS4
#	define GOB_PLATFORM_PS4 1
#elif defined(__PROSPERO__)
#	undef  GOB_PLATFORM_PS5
#	define GOB_PLATFORM_PS5 1
#elif  defined(__FreeBSD__)        \
	|| defined(__FreeBSD_kernel__) \
	|| defined(__NetBSD__)         \
	|| defined(__OpenBSD__)        \
	|| defined(__DragonFly__)
#	undef  GOB_PLATFORM_BSD
#	define GOB_PLATFORM_BSD 1
#elif defined(__GNU__)
#	undef  GOB_PLATFORM_HURD
#	define GOB_PLATFORM_HURD 1
#elif defined(__NX__)
#	undef  GOB_PLATFORM_NX
#	define GOB_PLATFORM_NX 1
#elif defined(__HAIKU__)
#	undef  GOB_PLATFORM_HAIKU
#	define GOB_PLATFORM_HAIKU 1
#endif //

#if !GOB_CRT_NONE
// https://sourceforge.net/p/predef/wiki/Libraries/
#	if defined(__BIONIC__)
#		undef  GOB_CRT_BIONIC
#		define GOB_CRT_BIONIC 1
#	elif defined(_MSC_VER)
#		undef  GOB_CRT_MSVC
#		define GOB_CRT_MSVC 1
#	elif defined(__GLIBC__)
#		undef  GOB_CRT_GLIBC
#		define GOB_CRT_GLIBC (__GLIBC__ * 10000 + __GLIBC_MINOR__ * 100)
#	elif defined(__MINGW32__) || defined(__MINGW64__)
#		undef  GOB_CRT_MINGW
#		define GOB_CRT_MINGW 1
#	elif defined(__apple_build_version__) || defined(__ORBIS__) || defined(__EMSCRIPTEN__) || defined(__llvm__) || defined(__HAIKU__)
#		undef  GOB_CRT_LIBCXX
#		define GOB_CRT_LIBCXX 1
#	elif GOB_PLATFORM_BSD
#		undef  GOB_CRT_BSD
#		define GOB_CRT_BSD 1
#	endif //

#	if !GOB_CRT_BIONIC \
	&& !GOB_CRT_BSD    \
	&& !GOB_CRT_GLIBC  \
	&& !GOB_CRT_LIBCXX \
	&& !GOB_CRT_MINGW  \
	&& !GOB_CRT_MSVC   \
	&& !GOB_CRT_NEWLIB
#		undef  GOB_CRT_NONE
#		define GOB_CRT_NONE 1
#	endif // GOB_CRT_*
#endif // !GOB_CRT_NONE

///
#define GOB_PLATFORM_POSIX (0   \
	||  GOB_PLATFORM_ANDROID    \
	||  GOB_PLATFORM_BSD        \
	||  GOB_PLATFORM_EMSCRIPTEN \
	||  GOB_PLATFORM_HAIKU      \
	||  GOB_PLATFORM_HURD       \
	||  GOB_PLATFORM_IOS        \
	||  GOB_PLATFORM_LINUX      \
	||  GOB_PLATFORM_NX         \
	||  GOB_PLATFORM_OSX        \
	||  GOB_PLATFORM_PS4        \
	||  GOB_PLATFORM_PS5        \
	||  GOB_PLATFORM_RPI        \
	)

///
#define GOB_PLATFORM_NONE !(0   \
	||  GOB_PLATFORM_ANDROID    \
	||  GOB_PLATFORM_BSD        \
	||  GOB_PLATFORM_EMSCRIPTEN \
	||  GOB_PLATFORM_HAIKU      \
	||  GOB_PLATFORM_HURD       \
	||  GOB_PLATFORM_IOS        \
	||  GOB_PLATFORM_LINUX      \
	||  GOB_PLATFORM_NX         \
	||  GOB_PLATFORM_OSX        \
	||  GOB_PLATFORM_PS4        \
	||  GOB_PLATFORM_PS5        \
	||  GOB_PLATFORM_RPI        \
	||  GOB_PLATFORM_WINDOWS    \
	||  GOB_PLATFORM_WINRT      \
	||  GOB_PLATFORM_XBOXONE    \
	)

///
#define GOB_PLATFORM_OS_CONSOLE  (0 \
	||  GOB_PLATFORM_NX             \
	||  GOB_PLATFORM_PS4            \
	||  GOB_PLATFORM_PS5            \
	||  GOB_PLATFORM_WINRT          \
	||  GOB_PLATFORM_XBOXONE        \
	)

///
#define GOB_PLATFORM_OS_DESKTOP  (0 \
	||  GOB_PLATFORM_BSD            \
	||  GOB_PLATFORM_HAIKU          \
	||  GOB_PLATFORM_HURD           \
	||  GOB_PLATFORM_LINUX          \
	||  GOB_PLATFORM_OSX            \
	||  GOB_PLATFORM_WINDOWS        \
	)

///
#define GOB_PLATFORM_OS_EMBEDDED (0 \
	||  GOB_PLATFORM_RPI            \
	)

///
#define GOB_PLATFORM_OS_MOBILE   (0 \
	||  GOB_PLATFORM_ANDROID        \
	||  GOB_PLATFORM_IOS            \
	)

///
#define GOB_PLATFORM_OS_WEB      (0 \
	||  GOB_PLATFORM_EMSCRIPTEN     \
	)

///
#if GOB_COMPILER_GCC
#	define GOB_COMPILER_NAME "GCC "       \
		GOB_STRINGIZE(__GNUC__) "."       \
		GOB_STRINGIZE(__GNUC_MINOR__) "." \
		GOB_STRINGIZE(__GNUC_PATCHLEVEL__)
#elif GOB_COMPILER_CLANG
#	define GOB_COMPILER_NAME "Clang "      \
		GOB_STRINGIZE(__clang_major__) "." \
		GOB_STRINGIZE(__clang_minor__) "." \
		GOB_STRINGIZE(__clang_patchlevel__)
#elif GOB_COMPILER_MSVC
#	if GOB_COMPILER_MSVC >= 1930 // Visual Studio 2022
#		define GOB_COMPILER_NAME "MSVC 17.0"
#	elif GOB_COMPILER_MSVC >= 1920 // Visual Studio 2019
#		define GOB_COMPILER_NAME "MSVC 16.0"
#	elif GOB_COMPILER_MSVC >= 1910 // Visual Studio 2017
#		define GOB_COMPILER_NAME "MSVC 15.0"
#	elif GOB_COMPILER_MSVC >= 1900 // Visual Studio 2015
#		define GOB_COMPILER_NAME "MSVC 14.0"
#	elif GOB_COMPILER_MSVC >= 1800 // Visual Studio 2013
#		define GOB_COMPILER_NAME "MSVC 12.0"
#	elif GOB_COMPILER_MSVC >= 1700 // Visual Studio 2012
#		define GOB_COMPILER_NAME "MSVC 11.0"
#	elif GOB_COMPILER_MSVC >= 1600 // Visual Studio 2010
#		define GOB_COMPILER_NAME "MSVC 10.0"
#	elif GOB_COMPILER_MSVC >= 1500 // Visual Studio 2008
#		define GOB_COMPILER_NAME "MSVC 9.0"
#	else
#		define GOB_COMPILER_NAME "MSVC"
#	endif //
#endif // GOB_COMPILER_

#if GOB_PLATFORM_ANDROID
#	define GOB_PLATFORM_NAME "Android " \
				GOB_STRINGIZE(GOB_PLATFORM_ANDROID)
#elif GOB_PLATFORM_BSD
#	define GOB_PLATFORM_NAME "BSD"
#elif GOB_PLATFORM_EMSCRIPTEN
#	define GOB_PLATFORM_NAME "asm.js "          \
		GOB_STRINGIZE(__EMSCRIPTEN_major__) "." \
		GOB_STRINGIZE(__EMSCRIPTEN_minor__) "." \
		GOB_STRINGIZE(__EMSCRIPTEN_tiny__)
#elif GOB_PLATFORM_HAIKU
#	define GOB_PLATFORM_NAME "Haiku"
#elif GOB_PLATFORM_HURD
#	define GOB_PLATFORM_NAME "Hurd"
#elif GOB_PLATFORM_IOS
#	define GOB_PLATFORM_NAME "iOS"
#elif GOB_PLATFORM_LINUX
#	define GOB_PLATFORM_NAME "Linux"
#elif GOB_PLATFORM_NONE
#	define GOB_PLATFORM_NAME "None"
#elif GOB_PLATFORM_NX
#	define GOB_PLATFORM_NAME "NX"
#elif GOB_PLATFORM_OSX
#	define GOB_PLATFORM_NAME "OSX"
#elif GOB_PLATFORM_PS4
#	define GOB_PLATFORM_NAME "PlayStation 4"
#elif GOB_PLATFORM_PS5
#	define GOB_PLATFORM_NAME "PlayStation 5"
#elif GOB_PLATFORM_RPI
#	define GOB_PLATFORM_NAME "RaspberryPi"
#elif GOB_PLATFORM_WINDOWS
#	define GOB_PLATFORM_NAME "Windows"
#elif GOB_PLATFORM_WINRT
#	define GOB_PLATFORM_NAME "WinRT"
#elif GOB_PLATFORM_XBOXONE
#	define GOB_PLATFORM_NAME "Xbox One"
#else
#	error "Unknown GOB_PLATFORM!"
#endif // GOB_PLATFORM_

#if GOB_CPU_ARM
#	define GOB_CPU_NAME "ARM"
#elif GOB_CPU_JIT
#	define GOB_CPU_NAME "JIT-VM"
#elif GOB_CPU_MIPS
#	define GOB_CPU_NAME "MIPS"
#elif GOB_CPU_PPC
#	define GOB_CPU_NAME "PowerPC"
#elif GOB_CPU_RISCV
#	define GOB_CPU_NAME "RISC-V"
#elif GOB_CPU_X86
#	define GOB_CPU_NAME "x86"
#endif // GOB_CPU_

#if GOB_CRT_BIONIC
#	define GOB_CRT_NAME "Bionic libc"
#elif GOB_CRT_BSD
#	define GOB_CRT_NAME "BSD libc"
#elif GOB_CRT_GLIBC
#	define GOB_CRT_NAME "GNU C Library"
#elif GOB_CRT_MSVC
#	define GOB_CRT_NAME "MSVC C Runtime"
#elif GOB_CRT_MINGW
#	define GOB_CRT_NAME "MinGW C Runtime"
#elif GOB_CRT_LIBCXX
#	define GOB_CRT_NAME "Clang C Library"
#elif GOB_CRT_NEWLIB
#	define GOB_CRT_NAME "Newlib"
#elif GOB_CRT_NONE
#	define GOB_CRT_NAME "None"
#else
#	error "Unknown GOB_CRT!"
#endif // GOB_CRT_

#if GOB_ARCH_32BIT
#	define GOB_ARCH_NAME "32-bit"
#elif GOB_ARCH_64BIT
#	define GOB_ARCH_NAME "64-bit"
#endif // GOB_ARCH_

#if defined(__cplusplus)
#	if   __cplusplus < GOB_LANGUAGE_CPP14
#		error "C++14 standard support is required to build."
#	elif __cplusplus < GOB_LANGUAGE_CPP17
#		define GOB_CPP_NAME "C++14"
#	elif __cplusplus < GOB_LANGUAGE_CPP20
#		define GOB_CPP_NAME "C++17"
#	elif __cplusplus < GOB_LANGUAGE_CPP23
#		define GOB_CPP_NAME "C++20"
#	else
// See: https://gist.github.com/bkaradzic/2e39896bc7d8c34e042b#orthodox-c
#		define GOB_CPP_NAME "C++WayTooModern"
#	endif // GOB_CPP_NAME
#else
#	define GOB_CPP_NAME "C++Unknown"
#endif // defined(__cplusplus)