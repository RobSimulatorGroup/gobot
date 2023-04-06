/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-4-5
*/

#include "gobot/editor/property_inspector/editor_property_math.hpp"

namespace gobot {

void EditorPropertyVector2::OnImGuiContent() {

    auto divided_width = ImGui::GetContentRegionAvail().x / 2.0f;

    float frame_height = ImGui::GetFrameHeight();
    ImVec2 button_size = { frame_height + 3.0f, frame_height };

    ImVec2 innerItemSpacing = ImGui::GetStyle().ItemInnerSpacing;
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, innerItemSpacing);

    // [0]
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0f, 1.0f, 1.0f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::Button("X", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        float kk = 0;
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        if (type_category_ == TypeCategory::Vector2i) {
            auto vector2i = property_data_model_->GetValue().convert<Vector2i>();
            if(ImGui::DragInt("##X", &vector2i.x(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                property_data_model_->SetValue(vector2i);
            }
        } else if (type_category_ == TypeCategory::Vector2f) {
            auto vector2f = property_data_model_->GetValue().convert<Vector2f>();
            if(ImGui::DragFloat("##X", &vector2f.x(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                property_data_model_->SetValue(vector2f);
            }
        } else if (type_category_ == TypeCategory::Vector2d) {
            auto vector2d = property_data_model_->GetValue().convert<Vector2d>();
            float x = vector2d.x();
            if(ImGui::DragFloat("##X", &x, 0.1f, 0.0f, 0.0f, "%.2f")) {
                vector2d.x() = x;
                property_data_model_->SetValue(vector2d);
            }
        }
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // [1]
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::Button("Y", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        if (type_category_ == TypeCategory::Vector2i) {
            auto vector2i = property_data_model_->GetValue().convert<Vector2i>();
            if(ImGui::DragInt("##Y", &vector2i.y(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                property_data_model_->SetValue(vector2i);
            }
        } else if (type_category_ == TypeCategory::Vector2f) {
            auto vector2f = property_data_model_->GetValue().convert<Vector2f>();
            if(ImGui::DragFloat("##Y", &vector2f.y(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                property_data_model_->SetValue(vector2f);
            }
        } else if (type_category_ == TypeCategory::Vector2d) {
            auto vector2d = property_data_model_->GetValue().convert<Vector2d>();
            float y = vector2d.y();
            if(ImGui::DragFloat("##Y", &y, 0.1f, 0.0f, 0.0f, "%.2f")) {
                vector2d.y() = y;
                property_data_model_->SetValue(vector2d);
            }
        }
        ImGui::PopStyleVar();
    }


    ImGui::PopStyleVar();

}

///////////////////////////////////////////////////////////////////

void EditorPropertyVector3::OnImGuiContent() {
    auto divided_width = ImGui::GetContentRegionAvail().x / 3.0f;

    float frame_height = ImGui::GetFrameHeight();
    ImVec2 button_size = { frame_height + 3.0f, frame_height };

    ImVec2 innerItemSpacing = ImGui::GetStyle().ItemInnerSpacing;
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, innerItemSpacing);

    // [0]
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0f, 1.0f, 1.0f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::Button("X", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        if (type_category_ == TypeCategory::Vector3i) {
            auto vector3i = property_data_model_->GetValue().convert<Vector3i>();
            if(ImGui::DragInt("##X", &vector3i.x(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                property_data_model_->SetValue(vector3i);
            }
        } else if (type_category_ == TypeCategory::Vector3f) {
            auto vector3f = property_data_model_->GetValue().convert<Vector3f>();
            if(ImGui::DragFloat("##X", &vector3f.x(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                property_data_model_->SetValue(vector3f);
            }
        } else if (type_category_ == TypeCategory::Vector3d) {
            auto vector3d = property_data_model_->GetValue().convert<Vector3d>();
            float x = vector3d.x();
            if(ImGui::DragFloat("##X", &x, 0.1f, 0.0f, 0.0f, "%.2f")) {
                vector3d.x() = x;
                property_data_model_->SetValue(vector3d);
            }
        }
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // [1]
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::Button("Y", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        if (type_category_ == TypeCategory::Vector3i) {
            auto vector3i = property_data_model_->GetValue().convert<Vector3i>();
            if(ImGui::DragInt("##Y", &vector3i.y(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                property_data_model_->SetValue(vector3i);
            }
        } else if (type_category_ == TypeCategory::Vector3f) {
            auto vector3f = property_data_model_->GetValue().convert<Vector3f>();
            if(ImGui::DragFloat("##Y", &vector3f.y(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                property_data_model_->SetValue(vector3f);
            }
        } else if (type_category_ == TypeCategory::Vector3d) {
            auto vector3d = property_data_model_->GetValue().convert<Vector3d>();
            float y = vector3d.y();
            if(ImGui::DragFloat("##Y", &y, 0.1f, 0.0f, 0.0f, "%.2f")) {
                vector3d.y() = y;
                property_data_model_->SetValue(vector3d);
            }
        }
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // [2]
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
        ImGui::Button("Z", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        if (type_category_ == TypeCategory::Vector3i) {
            auto vector3i = property_data_model_->GetValue().convert<Vector3i>();
            if(ImGui::DragInt("##Z", &vector3i.z(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                property_data_model_->SetValue(vector3i);
            }
        } else if (type_category_ == TypeCategory::Vector2f) {
            auto vector3f = property_data_model_->GetValue().convert<Vector3f>();
            if(ImGui::DragFloat("##Z", &vector3f.z(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                property_data_model_->SetValue(vector3f);
            }
        } else if (type_category_ == TypeCategory::Vector2d) {
            auto vector3d = property_data_model_->GetValue().convert<Vector3d>();
            float z = vector3d.z();
            if(ImGui::DragFloat("##Z", &z, 0.1f, 0.0f, 0.0f, "%.2f")) {
                vector3d.z() = z;
                property_data_model_->SetValue(vector3d);
            }
        }
        ImGui::PopStyleVar();
    }


    ImGui::PopStyleVar();

}

///////////////////////////////////////////////////////////////////

void EditorPropertyVector4::OnImGuiContent() {
    auto divided_width = ImGui::GetContentRegionAvail().x / 4.0f;

    float frame_height = ImGui::GetFrameHeight();
    ImVec2 button_size = { frame_height + 3.0f, frame_height };

    ImVec2 innerItemSpacing = ImGui::GetStyle().ItemInnerSpacing;
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, innerItemSpacing);

    // [0]
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0f, 1.0f, 1.0f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::Button("X", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        if (type_category_ == TypeCategory::Vector4i) {
            auto vector4i = property_data_model_->GetValue().convert<Vector4i>();
            if(ImGui::DragInt("##X", &vector4i.x(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                property_data_model_->SetValue(vector4i);
            }
        } else if (type_category_ == TypeCategory::Vector4f) {
            auto vector4f = property_data_model_->GetValue().convert<Vector4f>();
            if(ImGui::DragFloat("##X", &vector4f.x(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                property_data_model_->SetValue(vector4f);
            }
        } else if (type_category_ == TypeCategory::Vector4d) {
            auto vector4d = property_data_model_->GetValue().convert<Vector4d>();
            float x = vector4d.x();
            if(ImGui::DragFloat("##X", &x, 0.1f, 0.0f, 0.0f, "%.2f")) {
                vector4d.x() = x;
                property_data_model_->SetValue(vector4d);
            }
        }
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // [1]
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::Button("Y", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        if (type_category_ == TypeCategory::Vector4i) {
            auto vector4i = property_data_model_->GetValue().convert<Vector4i>();
            if(ImGui::DragInt("##Y", &vector4i.y(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                property_data_model_->SetValue(vector4i);
            }
        } else if (type_category_ == TypeCategory::Vector4f) {
            auto vector4f = property_data_model_->GetValue().convert<Vector4f>();
            if(ImGui::DragFloat("##Y", &vector4f.y(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                property_data_model_->SetValue(vector4f);
            }
        } else if (type_category_ == TypeCategory::Vector4d) {
            auto vector4d = property_data_model_->GetValue().convert<Vector4d>();
            float y = vector4d.y();
            if(ImGui::DragFloat("##Y", &y, 0.1f, 0.0f, 0.0f, "%.2f")) {
                vector4d.y() = y;
                property_data_model_->SetValue(vector4d);
            }
        }
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // [2]
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
        ImGui::Button("Z", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        if (type_category_ == TypeCategory::Vector4i) {
            auto vector4i = property_data_model_->GetValue().convert<Vector4i>();
            if(ImGui::DragInt("##Z", &vector4i.z(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                property_data_model_->SetValue(vector4i);
            }
        } else if (type_category_ == TypeCategory::Vector4f) {
            auto vector4f = property_data_model_->GetValue().convert<Vector4f>();
            if(ImGui::DragFloat("##Z", &vector4f.z(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                property_data_model_->SetValue(vector4f);
            }
        } else if (type_category_ == TypeCategory::Vector4d) {
            auto vector4d = property_data_model_->GetValue().convert<Vector4d>();
            float z = vector4d.z();
            if(ImGui::DragFloat("##Y", &z, 0.1f, 0.0f, 0.0f, "%.2f")) {
                vector4d.z() = z;
                property_data_model_->SetValue(vector4d);
            }
        }
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // [3]
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.54f, 0.17f, 0.89f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.9f, 0.9f, 0.9f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::Button("W", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        if (type_category_ == TypeCategory::Vector4i) {
            auto vector4i = property_data_model_->GetValue().convert<Vector4i>();
            if(ImGui::DragInt("##W", &vector4i.w(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                property_data_model_->SetValue(vector4i);
            }
        } else if (type_category_ == TypeCategory::Vector4f) {
            auto vector4f = property_data_model_->GetValue().convert<Vector4f>();
            if(ImGui::DragFloat("##W", &vector4f.w(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                property_data_model_->SetValue(vector4f);
            }
        } else if (type_category_ == TypeCategory::Vector4d) {
            auto vector4d = property_data_model_->GetValue().convert<Vector4d>();
            float w = vector4d.w();
            if(ImGui::DragFloat("##W", &w, 0.1f, 0.0f, 0.0f, "%.2f")) {
                vector4d.w() = w;
                property_data_model_->SetValue(vector4d);
            }
        }
        ImGui::PopStyleVar();
    }


    ImGui::PopStyleVar();

}


}
