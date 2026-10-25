#include "gobot/editor/imgui/tactile_panel.hpp"
#include "gobot/editor/editor.hpp"
#include "gobot/scene/tactile_sensor_3d.hpp"
#include "gobot/core/io/image.hpp"
#include "gobot/core/registration.hpp"
#include "gobot/rendering/rendering_server_globals.hpp"
#include "gobot/rendering/texture_storage.hpp"
#include <imgui.h>
#include <algorithm>
#include <cmath>

namespace gobot {
TactilePanel::TactilePanel() {
    SetName("TactilePanel");
    SetImGuiWindow("Tactile", "tactile");
}
TactilePanel::~TactilePanel() {
    if (texture_.IsValid() && RSG::texture_storage)
        RSG::texture_storage->TextureFree(texture_);
}
void TactilePanel::OnImGuiContent() {
    auto *editor = Editor::GetInstanceOrNull();
    std::vector<TactileSensor3D *> sensors;
    const auto visit = [&](auto &&self, Node *node) -> void {
        if (!node)
            return;
        if (auto *sensor = Object::PointerCastTo<TactileSensor3D>(node))
            sensors.push_back(sensor);
        for (int i = 0; i < node->GetChildCount(); ++i)
            self(self, node->GetChild(i));
    };
    if (editor)
        visit(visit, editor->GetActiveSceneRoot());
    const auto clear_preview = [&] {
        displayed_.reset();
        if (texture_.IsValid() && RSG::texture_storage)
            RSG::texture_storage->TextureFree(texture_);
        texture_ = RID{};
    };
    if (sensors.empty()) {
        clear_preview();
        ImGui::TextWrapped("Add a TactileSensor3D or open the GelSight Mini example.");
        return;
    }
    sensor_index_ = std::clamp(sensor_index_, 0, int(sensors.size()) - 1);
    if (ImGui::BeginCombo("Sensor", sensors[sensor_index_]->GetName().c_str())) {
        for (std::size_t i = 0; i < sensors.size(); ++i)
            if (ImGui::Selectable(sensors[i]->GetName().c_str(), int(i) == sensor_index_))
                sensor_index_ = int(i);
        ImGui::EndCombo();
    }
    auto frame = sensors[sensor_index_]->GetRuntimeFrame();
    if (!frame) {
        clear_preview();
        ImGui::TextWrapped("Press Play to preview tactile output.");
        return;
    }
    ImGui::Combo("View", &mode_,
                 "RGB\0Optical indentation\0Contact depth\0Pressure\0Shear magnitude\0Gel indentation\0");
    ImGui::Checkbox("Markers", &markers_);
    ImGui::SameLine();
    ImGui::Checkbox("Motion arrows", &arrows_);
    ImGui::Text("Time %.3f s   Frame %llu", frame->simulation_time, (unsigned long long)frame->tick);
    ImGui::Text("Force [N]  x %.4f  y %.4f  z %.4f", double(frame->total_force.x()),
                double(frame->total_force.y()), double(frame->total_force.z()));
    if (!RSG::texture_storage)
        return;
    if (frame != displayed_ || mode_ != last_mode_ || markers_ != last_markers_) {
        std::vector<std::uint8_t> pixels = markers_ ? frame->rgb_marker : frame->rgb;
        range_ = 0;
        if (mode_ != 0) {
            std::vector<float> values(frame->width * frame->height);
            for (std::size_t i = 0; i < values.size(); ++i) {
                values[i] = mode_ == 1   ? std::max(0.f, frame->optical_indentation[i]) * 1000.f
                            : mode_ == 5 ? std::max(0.f, frame->indentation[i]) * 1000.f
                            : mode_ == 2 ? frame->contact_depth[i] * 1000.f
                            : mode_ == 3
                                ? frame->normal_pressure[i]
                                : std::hypot(frame->shear_stress[2 * i], frame->shear_stress[2 * i + 1]);
                range_ = std::max(range_, values[i]);
            }
            for (std::size_t i = 0; i < values.size(); ++i) {
                const float t = range_ > 0 ? values[i] / range_ : 0;
                pixels[i * 3] = std::uint8_t(255 * std::clamp(3 * t, 0.f, 1.f));
                pixels[i * 3 + 1] = std::uint8_t(255 * std::clamp(3 * t - 1, 0.f, 1.f));
                pixels[i * 3 + 2] = std::uint8_t(255 * std::clamp(3 * t - 2, 0.f, 1.f));
            }
        }
        auto image = MakeRef<Image>(frame->width, frame->height, false, ImageFormat::RGB8, pixels);
        if (texture_.IsValid() && displayed_ &&
            (frame->width != displayed_->width || frame->height != displayed_->height)) {
            RSG::texture_storage->TextureFree(texture_);
            texture_ = RID{};
        }
        if (!texture_.IsValid()) {
            texture_ = RSG::texture_storage->TextureAllocate();
            RSG::texture_storage->Texture2DInitialize(texture_, image);
        } else
            RSG::texture_storage->TextureSetData(texture_, image);
        RSG::texture_storage->TextureSetLinearFilter(texture_, true);
        displayed_ = frame;
        last_mode_ = mode_;
        last_markers_ = markers_;
    }
    if (mode_ != 0)
        ImGui::Text("Scale 0 - %.4g %s", range_, (mode_ <= 2 || mode_ == 5) ? "mm" : "Pa");
    const float width = std::max(1.f, ImGui::GetContentRegionAvail().x);
    const ImVec2 size{width, width * frame->height / frame->width};
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::Image((ImTextureID)RSG::texture_storage->GetTextureNativeHandle(texture_), size);
    if (arrows_) {
        auto *draw = ImGui::GetWindowDrawList();
        const float scale = width / frame->width;
        for (std::size_t i = 0; i < frame->marker_positions.size(); ++i) {
            const Vector2 uv = frame->marker_positions[i], ref = uv - frame->marker_motion[i];
            const ImVec2 a{origin.x + float(ref.x()) * scale, origin.y + float(ref.y()) * scale};
            const ImVec2 b{origin.x + float(uv.x()) * scale, origin.y + float(uv.y()) * scale};
            draw->AddLine(a, b, IM_COL32(70, 255, 180, 255), 1.5f);
            draw->AddCircleFilled(b, 2.f, IM_COL32(70, 255, 180, 255));
        }
    }
}
} // namespace gobot
GOBOT_REGISTRATION { Class_<gobot::TactilePanel>("TactilePanel").constructor()(CtorAsRawPtr); }
