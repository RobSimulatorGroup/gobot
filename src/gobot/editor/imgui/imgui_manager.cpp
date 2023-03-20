/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-8
*/

#include "gobot/editor/imgui/imgui_manager.hpp"
#include "gobot/editor/imgui/imgui_utilities.hpp"
#include "gobot/rendering/imgui/imgui_impl_bgfx.hpp"
#include "gobot/rendering/default_view_id.hpp"
#include "gobot/drivers/sdl/sdl_window.hpp"
#include "gobot/scene/scene_tree.hpp"
#include "gobot/scene/window.hpp"
#include "gobot/platfom.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/log.hpp"
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_extension/fonts/RobotoRegular.inl"
#include "imgui_extension/fonts/RobotoBold.inl"
#include "imgui_extension/fonts/MaterialDesign.inl"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"
#include "imgui_extension/gizmos/ImGuizmo.h"

namespace gobot {

ImGuiManager* ImGuiManager::s_singleton = nullptr;

ImGuiManager::ImGuiManager()
{

    font_size_ = 14.0f;
    s_singleton = this;

    LOG_INFO("ImGui Version : {0}", IMGUI_VERSION);

    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows
    //io.ConfigViewportsNoAutoMerge = true;
    //io.ConfigViewportsNoTaskBarIcon = true;

    SetImGuiStyle();


    auto window = SceneTree::GetInstance()->GetRoot()->GetWindow();

    ImGui_Implbgfx_Init(IMGUI_VIEW_ID);
#if GOB_PLATFORM_WINDOWS
    ImGui_ImplSDL2_InitForD3D(window->GetSDL2Window());
#elif GOB_PLATFORM_OSX
    ImGui_ImplSDL2_InitForMetal(window->GetSDL2Window());
#elif GOB_PLATFORM_LINUX || GOB_PLATFORM_EMSCRIPTEN
    ImGui_ImplSDL2_InitForOpenGL(window->GetSDL2Window(), nullptr);
#endif

}

ImGuiManager::~ImGuiManager()
{
    s_singleton = nullptr;
    ImGui_ImplSDL2_Shutdown();
    ImGui_Implbgfx_Shutdown();
    ImGui::DestroyContext();
}

ImGuiManager* ImGuiManager::GetInstance()
{
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize ImGuiManager");
    return s_singleton;
}

void ImGuiManager::BeginFrame() {
    ImGui_Implbgfx_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();
    ImGuizmo::BeginFrame();
}

void ImGuiManager::EndFrame() {
    ImGui::Render();
    ImGui_Implbgfx_RenderDrawLists(ImGui::GetDrawData());
}

void ImGuiManager::SetImGuiStyle() {
    ImGuiIO& io = ImGui::GetIO();

    ImGui::StyleColorsDark();

    io.FontGlobalScale = 1.0f;

    ImFontConfig icons_config;
    icons_config.MergeMode   = false;
    icons_config.PixelSnapH  = true;
    icons_config.OversampleH = icons_config.OversampleV = 1;
    icons_config.GlyphMinAdvanceX                       = 4.0f;
    icons_config.SizePixels                             = 12.0f;

    static const ImWchar ranges[] = {
            0x0020,
            0x00FF,
            0x0400,
            0x044F,
            0,
    };

    io.Fonts->AddFontFromMemoryCompressedTTF(RobotoRegular_compressed_data, RobotoRegular_compressed_size, font_size_, &icons_config, ranges);
    AddIconFont();

    io.Fonts->AddFontFromMemoryCompressedTTF(RobotoBold_compressed_data, RobotoBold_compressed_size, font_size_ + 2.0f, &icons_config, ranges);

    io.Fonts->AddFontDefault();
    AddIconFont();

    io.Fonts->TexGlyphPadding = 1;
    for(int n = 0; n < io.Fonts->ConfigData.Size; n++) {
        ImFontConfig* font_config       = (ImFontConfig*)&io.Fonts->ConfigData[n];
        font_config->RasterizerMultiply = 1.0f;
    }

    ImGuiStyle& style = ImGui::GetStyle();

    style.WindowPadding     = ImVec2(5, 5);
    style.FramePadding      = ImVec2(4, 4);
    style.ItemSpacing       = ImVec2(6, 2);
    style.ItemInnerSpacing  = ImVec2(2, 2);
    style.IndentSpacing     = 6.0f;
    style.TouchExtraPadding = ImVec2(4, 4);

    style.ScrollbarSize = 10;

    style.WindowBorderSize = 0;
    style.ChildBorderSize  = 1;
    style.PopupBorderSize  = 3;
    style.FrameBorderSize  = 0.0f;

    const int roundingAmount = 2;
    style.PopupRounding      = roundingAmount;
    style.WindowRounding     = roundingAmount;
    style.ChildRounding      = 0;
    style.FrameRounding      = roundingAmount;
    style.ScrollbarRounding  = roundingAmount;
    style.GrabRounding       = roundingAmount;
    style.WindowMinSize      = ImVec2(200.0f, 200.0f);

#ifdef IMGUI_HAS_DOCK
    style.TabBorderSize = 1.0f;
    style.TabRounding   = roundingAmount; // + 4;

    if(ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding              = roundingAmount;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }
#endif

    ImGuiUtilities::SetTheme(ImGuiUtilities::Theme::Black);
}

void ImGuiManager::AddIconFont() {
    ImGuiIO& io = ImGui::GetIO();

    static const ImWchar icons_ranges[] = { ICON_MIN_MDI, ICON_MAX_MDI, 0 };
    ImFontConfig icons_config;
    // merge in icons from Font Awesome
    icons_config.MergeMode     = true;
    icons_config.PixelSnapH    = true;
    icons_config.GlyphOffset.y = 1.0f;
    icons_config.OversampleH = icons_config.OversampleV = 1;
    icons_config.GlyphMinAdvanceX                       = 4.0f;
    icons_config.SizePixels                             = 12.0f;

    io.Fonts->AddFontFromMemoryCompressedTTF(MaterialDesign_compressed_data, MaterialDesign_compressed_size, font_size_, &icons_config, icons_ranges);
}

}
