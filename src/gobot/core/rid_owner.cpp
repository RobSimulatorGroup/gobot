/*
 * The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * This file is created by Zikun Yu, 23-3-20
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gobot/core/rid_owner.hpp"

std::atomic<uint64_t> gobot::RID_AllocBase::base_id_{1};
