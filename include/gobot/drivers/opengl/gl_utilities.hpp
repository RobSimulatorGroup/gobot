/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-20
*/

#pragma once

#include "gobot_export.h"
#include "gobot/render/definitions.hpp"

namespace gobot {

class GOBOT_EXPORT GLUtilities {
public:
    static uint32_t FormatToGL(RHIFormat format, bool srgb = true);
    static uint32_t TextureWrapToGL(TextureWrap wrap);
    static uint32_t FormatToInternalFormat(uint32_t format);
    static uint32_t StencilTypeToGL(const StencilType type);

    static uint32_t RendererBufferToGL(uint32_t buffer);
    static uint32_t RendererBlendFunctionToGL(RendererBlendFunction function);
    static uint32_t DataTypeToGL(DataType dataType);
    static uint32_t DrawTypeToGL(DrawType drawType);
};



}
