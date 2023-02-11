///* The gobot is a robot simulation platform.
// * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
// * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
// * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
// * This file is created by Qiqi Wu, 23-1-14
//*/
//
//#pragma once
//
//#include "gobot/scene/node.hpp"
//
//#include <Eigen/Dense>
//#include "gobot/core/events/event.hpp"
//#include "gobot/graphics/RHI/graphics_context.hpp"
//
//namespace gobot {
//
//class GOBOT_EXPORT Window : public Node {
//    GOBCLASS(Window, Node);
//public:
//    using WindowHandle = void*;
//    using EventCallbackFn = std::function<void(Event &)>;
//
//    Window();
//
//    virtual ~Window();
//
//    [[nodiscard]] std::uint32_t GetWidth() const;
//
//    [[nodiscard]] std::uint32_t GetHeight() const;
//
//    [[nodiscard]] String GetTitle() const;
//
//    void SetWindowTitle(const String &title);
//
//    [[nodiscard]] float GetDPIScale() const { return 1.0f; }
//
//    [[nodiscard]] FORCE_INLINE bool GetVSync() const { return data_.v_sync_; };
//
//    void ToggleVSync();
//
//    void SetVSync(bool v_sync);
//
//    [[nodiscard]] float GetScreenRatio() const;
//
//    [[nodiscard]] WindowHandle GetNativeWindowHandle() const;
//
//    [[nodiscard]] bool IsMaximized();
//
//    void Minimize();
//
//    void Maximize();
//
//    void Restore();
//
//    void SetWindowFocus(bool focus) { data_.window_focus_ = focus; }
//
//    [[nodiscard]] bool GetWindowFocus() const { return data_.window_focus_; }
//
//    void SetIcon(const std::string &file_path, const std::string &small_icon_file_path = "");
//
//    void OnUpdate();
//
//    void UpdateCursorImGui();
//
//    void HideMouse(bool hide);
//
//    void SetMousePosition(const Eigen::Vector2f &pos);
//
//    void SetEventCallback(const EventCallbackFn &callback);
//
//private:
//    struct Data {
//        bool initialised_;
//        bool v_sync_;
//        bool window_focus_;
//        float dpi_scale_;
//        uint32_t width;
//        uint32_t height_;
//        String title_;
//        RenderAPI render_api_;
//
//        EventCallbackFn event_callback_;
//
//        GLFWwindow* native_handle_ = nullptr;
//        Window* holder_{nullptr};
//    };
//    Data data_;
//};
//
//}