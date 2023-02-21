/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-2-12
*/

#pragma once

#include "gobot/graphics/RHI/imgui_renderer.hpp"

namespace gobot {

class GOBOT_EXPORT GLIMGUIRenderer : public IMGUIRenderer
{
public:
    GLIMGUIRenderer(std::uint32_t width, std::uint32_t height, bool clear_screen);

    ~GLIMGUIRenderer();

    void Init() override;

    void NewFrame() override;

    void Render(CommandBuffer* commandBuffer) override;

    void OnResize(std::uint32_t width, std::uint32_t height) override;

    bool Implemented() const override { return true; }

    void RebuildFontTexture() override;

    static void MakeDefault();

protected:
    static IMGUIRenderer* CreateFuncGL(std::uint32_t width, std::uint32_t height, bool clear_screen);

private:
    void* window_handle_;
    bool clear_screen_;
};

}
