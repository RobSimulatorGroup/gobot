/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-18
*/

#pragma once

#include "gobot/core/object.hpp"

namespace gobot {

struct VersionInfo {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    String commit;
};

class Engine : public Object {
    GOBCLASS(Engine, Object)
public:
    Engine();

    virtual ~Engine();

    static Engine* GetInstance();

    [[nodiscard]] FORCE_INLINE double GetTimeScale() const { return time_scale_; }

    FORCE_INLINE void SetTimeScale(double time_scale) { time_scale_ = time_scale; }

    [[nodiscard]] VersionInfo GetVersionInfo() const;


private:
    static Engine* s_singleton;

    double time_scale_;
};

}
