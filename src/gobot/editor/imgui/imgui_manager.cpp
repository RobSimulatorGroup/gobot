/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-3-8
*/

#include "gobot/editor/imgui/imgui_manager.hpp"
#include "gobot/editor/imgui/imgui_utilities.hpp"
#include "gobot/drivers/sdl/sdl_window.hpp"
#include "gobot/scene/scene_tree.hpp"
#include "gobot/scene/window.hpp"
#include "gobot/platfom.hpp"
#include "gobot/error_macros.hpp"
#include "gobot/log.hpp"
#include "gobot/rendering/render_server.hpp"
#include "imgui.h"
#include "imgui_extension/fonts/RobotoRegular.inl"
#include "imgui_extension/fonts/RobotoBold.inl"
#include "imgui_extension/fonts/MaterialDesign.inl"
#include "imgui_extension/icon_fonts/icons_material_design_icons.h"
#include "imgui_extension/gizmos/ImGuizmo.h"
#include "gobot/drivers/opengl/imgui_renderer.hpp"

#include <SDL.h>
#include <algorithm>
#include <cstdlib>

namespace gobot {

ImGuiManager* ImGuiManager::s_singleton = nullptr;

ImGuiManager::ImGuiManager()
{
    ImGui::CreateContext();
    ui_scale_ = DetectUIScale();
    font_size_ = 18.0f * ui_scale_;
    s_singleton = this;

    ImGui::StyleColorsDark();
    SetImGuiStyle();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows
    //io.ConfigViewportsNoAutoMerge = true;
    //io.ConfigViewportsNoTaskBarIcon = true;

    if (RS::GetInstance()->GetRendererType() == RendererType::OpenGL46) {
        imgui_renderer_ = std::make_unique<opengl::ImGuiGLRenderer>();
    } else {
        // TODO(wqq)
    }

    auto window = SceneTree::GetInstance()->GetRoot()->GetWindow();
    imgui_renderer_->Init(window->GetSDL2Window());

    LOG_INFO("ImGui Version : {0}, UI scale: {1}, font size: {2}", IMGUI_VERSION, ui_scale_, font_size_);
}

ImGuiManager::~ImGuiManager()
{
    s_singleton = nullptr;
    imgui_renderer_.reset();
    ImGui::DestroyContext();
}

ImGuiManager* ImGuiManager::GetInstance()
{
    ERR_FAIL_COND_V_MSG(s_singleton == nullptr, nullptr, "Must call this after initialize ImGuiManager");
    return s_singleton;
}

void ImGuiManager::BeginFrame() {
    imgui_renderer_->NewFrame();
    ImGui::NewFrame();
    ImGuizmo::BeginFrame();
}

void ImGuiManager::EndFrame() {
    imgui_renderer_->Render();
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
    style.PopupBorderSize  = 1;
    style.FrameBorderSize  = 0.0f;

    const int rounding_amount = 2;
    style.PopupRounding      = rounding_amount;
    style.WindowRounding     = rounding_amount;
    style.ChildRounding      = 0;
    style.FrameRounding      = rounding_amount;
    style.ScrollbarRounding  = rounding_amount;
    style.GrabRounding       = rounding_amount;
    style.WindowMinSize      = ImVec2(200.0f, 200.0f);
    style.ScaleAllSizes(ui_scale_);

#ifdef IMGUI_HAS_DOCK
    style.TabBorderSize = 1.0f;
    style.TabRounding   = rounding_amount; // + 4;

    if(ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding              = rounding_amount;
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

float ImGuiManager::DetectUIScale() const {
    if (const char* env_scale = std::getenv("GOBOT_UI_SCALE")) {
        char* end = nullptr;
        const float parsed = std::strtof(env_scale, &end);
        if (end != env_scale && parsed > 0.0f) {
            return std::clamp(parsed, 0.75f, 4.0f);
        }
    }

    auto* window = SceneTree::GetInstance()->GetRoot()->GetWindow();
    SDL_Window* sdl_window = window ? window->GetSDL2Window() : nullptr;
    if (!sdl_window) {
        return 1.0f;
    }

    float dpi_scale = 1.0f;
    const int display_index = SDL_GetWindowDisplayIndex(sdl_window);
    float ddpi = 0.0f;
    if (display_index >= 0 && SDL_GetDisplayDPI(display_index, &ddpi, nullptr, nullptr) == 0 && ddpi > 0.0f) {
        dpi_scale = ddpi / 96.0f;
    }

    int window_width = 0;
    int window_height = 0;
    int drawable_width = 0;
    int drawable_height = 0;
    SDL_GetWindowSize(sdl_window, &window_width, &window_height);
    SDL_GL_GetDrawableSize(sdl_window, &drawable_width, &drawable_height);
    float framebuffer_scale = 1.0f;
    if (window_width > 0 && window_height > 0 && drawable_width > 0 && drawable_height > 0) {
        framebuffer_scale = std::max(static_cast<float>(drawable_width) / static_cast<float>(window_width),
                                     static_cast<float>(drawable_height) / static_cast<float>(window_height));
    }

    return std::clamp(std::max(dpi_scale, framebuffer_scale), 1.0f, 3.0f);
}

}
