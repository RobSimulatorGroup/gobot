/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-6-27
*/

#pragma once

namespace gobot {

class ShaderStorage {
public:
    virtual ~ShaderStorage() {};

    virtual RID ShaderAllocate() = 0;

    virtual void Initialize(RID shader) = 0;

    virtual void ShaderSetCode(RID shader, String code) = 0;

    virtual String ShaderGetCode(RID shader) = 0;

    virtual void ShaderFree(RID shader) = 0;
};


}
