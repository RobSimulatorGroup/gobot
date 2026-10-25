#pragma once
#include "gobot/scene/imgui_window.hpp"
#include "gobot/core/rid.hpp"
#include "gobot/core/tactile_sensor_frame.hpp"
namespace gobot {
class GOBOT_EXPORT TactilePanel : public ImGuiWindow {
    GOBCLASS(TactilePanel, ImGuiWindow)
  public:
    TactilePanel();
    ~TactilePanel() override;
    void OnImGuiContent() override;

  private:
    RID texture_;
    std::shared_ptr<const TactileSensorFrame> displayed_;
    int mode_{0}, sensor_index_{0}, last_mode_{-1};
    bool markers_{true}, arrows_{false}, last_markers_{false};
    float range_{0};
};
} // namespace gobot
