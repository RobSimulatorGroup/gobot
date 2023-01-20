/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-1-14
*/

#pragma once

#include "gobot/log.hpp"
#include "gobot/core/marcos.hpp"

#define ERR_FAIL_COND(cond)                                                                    \
	if (cond) [[unlikely]] {                                                                   \
        LOG_ERROR("Failed condition {}.", GOB_STRINGIFY(cond));                                \
		return;                                                                                \
	} else                                                                                     \
		((void)0)


#define ERR_FAIL_COND_V(cond, ret)                                                              \
	if (cond) [[unlikely]] {                                                                    \
        LOG_ERROR("Failed condition {}. Return: {}", GOB_STRINGIFY(cond), GOB_STRINGIFY(ret));  \
		return ret;                                                                             \
	} else                                                                                      \
		((void)0)


#define ERR_FAIL_COND_MSG(cond, msg)                                                            \
	if (cond) [[unlikely]] {                                                                    \
        LOG_ERROR("Failed condition {}. {}", GOB_STRINGIFY(cond), GOB_STRINGIFY(msg));          \
		return;                                                                                 \
	} else                                                                                      \
		((void)0)


/**
 * Try using `ERR_FAIL_INDEX_V_MSG`.
 * Only use this macro if there is no sensible error message.
 *
 * Ensures an integer index `m_index` is less than `m_size` and greater than or equal to 0.
 * If not, the current function returns `m_retval`.
 */
#define ERR_FAIL_INDEX_MSG(index, size, msg)                                                    \
    if ((index) < 0 || (index) >= (size)) [[unlikely]] {                                        \
        LOG_ERROR("Invalid index {} out of {}. {}", index, size, GOB_STRINGIFY(msg));           \
        return;                                                                                 \
    } else                                                                                      \
        ((void)0)


/**
 * Ensures an integer index `m_index` is less than `m_size` and greater than or equal to 0.
 * If not, prints `m_msg` and the current function returns `m_retval`.
 */
#define ERR_FAIL_INDEX_V(index, size, ret)                                                      \
    if ((index) < 0 || (index) >= (size)) [[unlikely]] {                                        \
        LOG_ERROR("Invalid index {} out of {}, return {}.", idx, size, GOB_STRINGIFY(ret));     \
        return ret;                                                                             \
    } else                                                                                      \
        ((void)0)
