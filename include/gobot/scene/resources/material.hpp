/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-12-21
*/

#pragma once

#include "gobot/core/io/resource.hpp"

namespace gobot {

// Material is a base Resource used for coloring and shading geometry.
class GOBOT_EXPORT Material : public Resource {
    GOBCLASS(Material, Resource)
public:
    Material();

private:


};


class GOBOT_EXPORT Material3D: public Material {
    GOBCLASS(Material3D, Material)
public:
    Material3D();

    void SetAlbedo(const Color &albedo);

    Color GetAlbedo() const;

private:
    Color albedo_;
};

}
