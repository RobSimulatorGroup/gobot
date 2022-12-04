/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 22-11-20
*/

#include "gobot/core/io/resource.hpp"

#include "gobot/core/registration.hpp"

namespace gobot {

Resource::Resource() {

}

void Resource::SetPath(const String &path) {

}

String Resource::GetPath() const {
    return path_cache_;
}

void Resource::SetName(const String &name) {
    name_ = name;
    Q_EMIT resourceChanged();
}

String Resource::GetName() const {
    return name_;
}

void Resource::SetResourceUuid(const QUuid &uuid) {
    uuid_ = uuid;
}

Uuid Resource::GetResourceUuid() const {
    return uuid_;
}


}

GOBOT_REGISTRATION {
    gobot::Class_<gobot::Resource>("gobot::core::Resource");

};
