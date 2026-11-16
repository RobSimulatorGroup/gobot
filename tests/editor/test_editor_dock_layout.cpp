#include "gobot/editor/imgui/default_dock_layout.hpp"
#include <gtest/gtest.h>

namespace {

class EditorDockLayoutTest : public testing::Test {
protected:
    void SetUp() override {
        context_ = ImGui::CreateContext();
        ImGui::GetIO().IniFilename = nullptr;
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    }

    void TearDown() override { ImGui::DestroyContext(context_); }

    ImGuiDockNode* Dock(const char* name) {
        const auto* settings = ImGui::FindWindowSettingsByID(ImHashStr(name));
        return settings ? ImGui::DockBuilderGetNode(settings->DockId) : nullptr;
    }

    ImGuiContext* context_ = nullptr;
    const ImGuiID dockspace_ = ImHashStr("DefaultLayoutTest");
};

TEST_F(EditorDockLayoutTest, PhysicsAndTactileShareTheBottomRightDock) {
    gobot::BuildDefaultEditorDockLayout(dockspace_, ImVec2(1600, 900));
    auto* physics = Dock("###physics");
    auto* tactile = Dock("###tactile");
    auto* inspector = Dock("###inspector");
    auto* viewport = Dock("###scene_view3d");
    ASSERT_NE(physics, nullptr);
    ASSERT_NE(inspector, nullptr);
    ASSERT_NE(viewport, nullptr);
    EXPECT_EQ(physics, tactile);
    EXPECT_FLOAT_EQ(physics->Pos.x, inspector->Pos.x);
    EXPECT_GT(physics->Pos.y, inspector->Pos.y);
    EXPECT_GT(physics->Pos.x, viewport->Pos.x);
    EXPECT_GT(physics->Size.y, inspector->Size.y);
    EXPECT_EQ(physics->SelectedTabId, ImHashStr("###physics"));
    EXPECT_EQ(ImGui::FindWindowSettingsByID(ImHashStr("###physics"))->DockOrder, 0);
    EXPECT_EQ(ImGui::FindWindowSettingsByID(ImHashStr("###tactile"))->DockOrder, 1);
}

TEST_F(EditorDockLayoutTest, ResetRestoresTheDefaultSimulationDock) {
    gobot::BuildDefaultEditorDockLayout(dockspace_, ImVec2(1600, 900));
    ImGui::DockBuilderDockWindow("###physics", Dock("###scene_view3d")->ID);
    EXPECT_NE(Dock("###physics"), Dock("###tactile"));
    gobot::BuildDefaultEditorDockLayout(dockspace_, ImVec2(1600, 900));
    ASSERT_NE(Dock("###physics"), nullptr);
    ASSERT_NE(Dock("###inspector"), nullptr);
    EXPECT_EQ(Dock("###physics"), Dock("###tactile"));
    EXPECT_GT(Dock("###physics")->Pos.y, Dock("###inspector")->Pos.y);
    EXPECT_EQ(Dock("###physics")->SelectedTabId, ImHashStr("###physics"));
}

} // namespace
