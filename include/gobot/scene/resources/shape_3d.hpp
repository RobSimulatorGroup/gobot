/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-1-13
*/

#include "gobot/core/io/resource.hpp"

namespace gobot {


// Shape3D is for collision checker
class GOBOT_EXPORT Shape3D : public Resource {
    GOBCLASS(Shape3D, Resource)
public:
    Shape3D();

    ~Shape3D();

};


}

