/* The gobot is a robot simulation platform. 
 * Copyright(c) 2021-2023, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Qiqi Wu, 23-4-5
*/

#include "gobot/editor/property_inspector/editor_property_math.hpp"

namespace gobot {

void EditorPropertyVector2::OnImGuiContent() {
    if (type_category_ == TypeCategory::Vector2i) {
        data_ = property_data_model_->GetValue().convert<Vector2i>();
    } else if (type_category_ == TypeCategory::Vector2f) {
        data_ = property_data_model_->GetValue().convert<Vector2f>();
    } else if (type_category_ == TypeCategory::Vector2d) {
        data_ = property_data_model_->GetValue().convert<Vector2d>();
    }

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
        ImGui::Button("x", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Vector2i>) {
                if(ImGui::DragInt("##X", &arg.x(), 0.1f, 0.0f, 0.0f)) {
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Vector2f>) {
                if(ImGui::DragFloat("##X", &arg.x(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Vector2d>) {
                float x = arg.x();
                if(ImGui::DragFloat("##X", &x, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg.x() = x;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
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
        ImGui::Button("y", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Vector2i>) {
                if(ImGui::DragInt("##Y", &arg.y(), 0.1f, 0.0f, 0.0f)) {
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Vector2f>) {
                if(ImGui::DragFloat("##Y", &arg.y(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Vector2d>) {
                float y = arg.y();
                if(ImGui::DragFloat("##Y", &y, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg.y() = y;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::PopStyleVar();

}

///////////////////////////////////////////////////////////////////

void EditorPropertyVector3::OnImGuiContent() {
    if (type_category_ == TypeCategory::Vector3i) {
        data_ = property_data_model_->GetValue().convert<Vector3i>();
    } else if (type_category_ == TypeCategory::Vector3f) {
        data_ = property_data_model_->GetValue().convert<Vector3f>();
    } else if (type_category_ == TypeCategory::Vector3d) {
        data_ = property_data_model_->GetValue().convert<Vector3d>();
    }

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
        ImGui::Button("x", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Vector3i>) {
                if(ImGui::DragInt("##X", &arg.x(), 0.1f, 0.0f, 0.0f)) {
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Vector3f>) {
                if(ImGui::DragFloat("##X", &arg.x(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Vector3d>) {
                float x = arg.x();
                if(ImGui::DragFloat("##X", &x, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg.x() = x;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
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
        ImGui::Button("y", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Vector3i>) {
                if(ImGui::DragInt("##Y", &arg.y(), 0.1f, 0.0f, 0.0f)) {
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Vector3f>) {
                if(ImGui::DragFloat("##Y", &arg.y(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Vector3d>) {
                float y = arg.y();
                if(ImGui::DragFloat("##Y", &y, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg.y() = y;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
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
        ImGui::Button("z", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Vector3i>) {
                if(ImGui::DragInt("##Z", &arg.z(), 0.1f, 0.0f, 0.0f)) {
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Vector3f>) {
                if(ImGui::DragFloat("##Z", &arg.z(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Vector3d>) {
                float z = arg.z();
                if(ImGui::DragFloat("##Z", &z, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg.z() = z;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::PopStyleVar();

}

///////////////////////////////////////////////////////////////////

void EditorPropertyVector4::OnImGuiContent() {
    if (type_category_ == TypeCategory::Vector4i) {
        data_ = property_data_model_->GetValue().convert<Vector4i>();
    } else if (type_category_ == TypeCategory::Vector4f) {
        data_ = property_data_model_->GetValue().convert<Vector4f>();
    } else if (type_category_ == TypeCategory::Vector4d) {
        data_ = property_data_model_->GetValue().convert<Vector4d>();
    }

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
        ImGui::Button("x", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Vector4i>) {
                if(ImGui::DragInt("##X", &arg.x(), 0.1f, 0.0f, 0.0f)) {
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Vector4f>) {
                if(ImGui::DragFloat("##X", &arg.x(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Vector4d>) {
                float x = arg.x();
                if(ImGui::DragFloat("##X", &x, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg.x() = x;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
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
        ImGui::Button("y", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Vector4i>) {
                if(ImGui::DragInt("##Y", &arg.y(), 0.1f, 0.0f, 0.0f)) {
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Vector4f>) {
                if(ImGui::DragFloat("##Y", &arg.y(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Vector4d>) {
                float y = arg.y();
                if(ImGui::DragFloat("##Y", &y, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg.y() = y;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
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
        ImGui::Button("z", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Vector4i>) {
                if(ImGui::DragInt("##Z", &arg.z(), 0.1f, 0.0f, 0.0f)) {
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Vector4f>) {
                if(ImGui::DragFloat("##Z", &arg.z(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Vector4d>) {
                float z = arg.z();
                if(ImGui::DragFloat("##Z", &z, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg.z() = z;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
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
        ImGui::Button("w", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Vector4i>) {
                if(ImGui::DragInt("##W", &arg.w(), 0.1f, 0.0f, 0.0f)) {
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Vector4f>) {
                if(ImGui::DragFloat("##W", &arg.w(), 0.1f, 0.0f, 0.0f, "%.2f")) {
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Vector4d>) {
                float w = arg.w();
                if(ImGui::DragFloat("##W", &w, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg.w() = w;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::PopStyleVar();

}

///////////////////////////////////////////////////////////////////

void EditorPropertyQuaternion::OnImGuiContent() {
    if (type_category_ == TypeCategory::Quaternionf) {
        data_ = property_data_model_->GetValue().convert<Quaternionf>();
    } else if (type_category_ == TypeCategory::Quaterniond) {
        data_ = property_data_model_->GetValue().convert<Quaterniond>();
    }

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
        if (ImGui::Button("qx", button_size)) {
            ImGui::OpenPopup("Edit Mode");
        }
        if (ImGui::BeginPopup("Edit Mode")) {
            bool last_edit_mode = free_edit_mode_;
            if (ImGui::Checkbox("Edit Free ", &free_edit_mode_) && free_edit_mode_ == true) {
                LOG_INFO("Starting quaternion free edit mode, you can edit quaternion without normalized. You can make quaternion normalize when you close Free Edit Mode");
            }
            if (last_edit_mode == true && free_edit_mode_ == false) {
                std::visit([this](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, Quaternionf>) {
                        arg.normalize();
                        property_data_model_->SetValue(arg);
                    } else if constexpr (std::is_same_v<T, Quaterniond>) {
                        arg.normalize();
                        property_data_model_->SetValue(arg);
                    }
                }, data_);
            }
            ImGui::EndPopup();
        }

        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Quaternionf>) {
                if(ImGui::DragFloat("##QX", &arg.x(), 0.1f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    if (!free_edit_mode_) {
                        arg.normalize();
                    }
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Quaterniond>) {
                float x = arg.x();
                if(ImGui::DragFloat("##QX", &x, 0.1f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg.x() = x;
                    if (!free_edit_mode_) {
                        arg.normalize();
                    }
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
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
        ImGui::Button("qy", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Quaternionf>) {
                if(ImGui::DragFloat("##QY", &arg.y(), 0.1f, -1.0f, 1.0f, "%.4f", ImGuiSliderFlags_AlwaysClamp)) {
                    if (!free_edit_mode_) {
                        arg.normalize();
                    }
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Quaterniond>) {
                float y = arg.y();
                if(ImGui::DragFloat("##QY", &y, 0.1f, -1.0f, 1.0f, "%.4f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg.y() = y;
                    if (!free_edit_mode_) {
                        arg.normalize();
                    }
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
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
        ImGui::Button("qz", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Quaternionf>) {
                if(ImGui::DragFloat("##QZ", &arg.z(), 0.1f, -1.0f, 1.0f, "%.4f", ImGuiSliderFlags_AlwaysClamp)) {
                    if (!free_edit_mode_) {
                        arg.normalize();
                    }
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Quaterniond>) {
                float z = arg.z();
                if(ImGui::DragFloat("##QZ", &z, 0.1f, -1.0f, 1.0f, "%.4f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg.z() = z;
                    if (!free_edit_mode_) {
                        arg.normalize();
                    }
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
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
        ImGui::Button("qw", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Quaternionf>) {
                if(ImGui::DragFloat("##QW", &arg.w(), -1.0f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    if (!free_edit_mode_) {
                        arg.normalize();
                    }
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Quaterniond>) {
                float w = arg.w();
                if(ImGui::DragFloat("##QW", &w, 0.1f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg.w() = w;
                    if (!free_edit_mode_) {
                        arg.normalize();
                    }
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::PopStyleVar();
}

////////////////////////////////////////////////////////////////////

void EditorPropertyMatrix2::OnImGuiContent() {
    if (type_category_ == TypeCategory::Matrix2i) {
        data_ = property_data_model_->GetValue().convert<Matrix2i>();
    } else if (type_category_ == TypeCategory::Matrix2f) {
        data_ = property_data_model_->GetValue().convert<Matrix2f>();
    } else if (type_category_ == TypeCategory::Matrix2d) {
        data_ = property_data_model_->GetValue().convert<Matrix2d>();
    }

    auto divided_width = ImGui::GetContentRegionAvail().x / 2.0f;

    float frame_height = ImGui::GetFrameHeight();
    ImVec2 button_size = { frame_height + 3.0f, frame_height };

    ImVec2 innerItemSpacing = ImGui::GetStyle().ItemInnerSpacing;
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, innerItemSpacing);

    // XX
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0f, 1.0f, 1.0f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::Button("xx", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Matrix2i>) {
                int xx = arg(0, 0);
                if(ImGui::DragInt("##XX", &xx, 0.1f, 0.0f, 0.0f)) {
                    arg(0, 0) = xx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix2f>) {
                float xx = arg(0, 0);
                if(ImGui::DragFloat("##XX", &xx, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(0, 0) = xx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix2d>) {
                float xx = arg(0, 0);
                if(ImGui::DragFloat("##XX", &xx, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(0, 0) = xx;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // XY
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::Button("xy", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Matrix2i>) {
                int xy = arg(0, 1);
                if(ImGui::DragInt("##XY", &xy, 0.1f, 0.0f, 0.0f)) {
                    arg(0, 1) = xy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix2f>) {
                float xy = arg(0, 1);
                if(ImGui::DragFloat("##XY", &xy, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(0, 1) = xy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix2d>) {
                float xy = arg(0, 1);
                if(ImGui::DragFloat("##XY", &xy, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(0, 1) = xy;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    // YX
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0f, 1.0f, 1.0f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::Button("yx", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Matrix2i>) {
                int yz = arg(1, 0);
                if(ImGui::DragInt("##YX", &yz, 0.1f, 0.0f, 0.0f)) {
                    arg(1, 0) = yz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix2f>) {
                float yz = arg(1, 0);
                if(ImGui::DragFloat("##YX", &yz, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(1, 0) = yz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix2d>) {
                float yz = arg(1, 0);
                if(ImGui::DragFloat("##XY", &yz, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(1, 0) = yz;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // YY
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::Button("yy", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Matrix2i>) {
                int yy = arg(1, 1);
                if(ImGui::DragInt("##YY", &yy, 0.1f, 0.0f, 0.0f)) {
                    arg(1, 1) = yy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix2f>) {
                float yy = arg(1, 1);
                if(ImGui::DragFloat("##YY", &yy, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(1, 1) = yy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix2d>) {
                float yy = arg(1, 1);
                if(ImGui::DragFloat("##YY", &yy, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(1, 1) = yy;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::PopStyleVar();

}

/////////////////////////////////////////////////////////////////////

void EditorPropertyMatrix3::OnImGuiContent() {
    if (type_category_ == TypeCategory::Matrix3i) {
        data_ = property_data_model_->GetValue().convert<Matrix3i>();
    } else if (type_category_ == TypeCategory::Matrix3f) {
        data_ = property_data_model_->GetValue().convert<Matrix3f>();
    } else if (type_category_ == TypeCategory::Matrix3d) {
        data_ = property_data_model_->GetValue().convert<Matrix3d>();
    }

    auto divided_width = ImGui::GetContentRegionAvail().x / 3.0f;

    float frame_height = ImGui::GetFrameHeight();
    ImVec2 button_size = { frame_height + 3.0f, frame_height };

    ImVec2 innerItemSpacing = ImGui::GetStyle().ItemInnerSpacing;
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, innerItemSpacing);

    // XX
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0f, 1.0f, 1.0f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::Button("xx", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Matrix3i>) {
                int xx = arg(0, 0);
                if(ImGui::DragInt("##XX", &xx, 0.1f, 0.0f, 0.0f)) {
                    arg(0, 0) = xx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix3f>) {
                float xx = arg(0, 0);
                if(ImGui::DragFloat("##XX", &xx, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(0, 0) = xx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix3d>) {
                float xx = arg(0, 0);
                if(ImGui::DragFloat("##XX", &xx, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(0, 0) = xx;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // XY
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::Button("xy", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Matrix3i>) {
                int xy = arg(0, 1);
                if(ImGui::DragInt("##XY", &xy, 0.1f, 0.0f, 0.0f)) {
                    arg(0, 1) = xy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix3f>) {
                float xy = arg(0, 1);
                if(ImGui::DragFloat("##XY", &xy, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(0, 1) = xy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix3d>) {
                float xy = arg(0, 1);
                if(ImGui::DragFloat("##XY", &xy, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(0, 1) = xy;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // XZ
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
        ImGui::Button("xz", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Matrix3i>) {
                int xz = arg(0, 2);
                if(ImGui::DragInt("##XZ", &xz, 0.1f, 0.0f, 0.0f)) {
                    arg(0, 2) = xz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix3f>) {
                float xz = arg(0, 2);
                if(ImGui::DragFloat("##XZ", &xz, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(0, 2) = xz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix3d>) {
                float xz = arg(0, 2);
                if(ImGui::DragFloat("##XZ", &xz, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(0, 1) = xz;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    // YX
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0f, 1.0f, 1.0f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::Button("yx", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Matrix3i>) {
                int yz = arg(1, 0);
                if(ImGui::DragInt("##YX", &yz, 0.1f, 0.0f, 0.0f)) {
                    arg(1, 0) = yz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix3f>) {
                float yz = arg(1, 0);
                if(ImGui::DragFloat("##YX", &yz, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(1, 0) = yz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix3d>) {
                float yz = arg(1, 0);
                if(ImGui::DragFloat("##YX", &yz, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(1, 0) = yz;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // YY
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::Button("yy", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Matrix3i>) {
                int yy = arg(1, 1);
                if(ImGui::DragInt("##YY", &yy, 0.1f, 0.0f, 0.0f)) {
                    arg(1, 1) = yy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix3f>) {
                float yy = arg(1, 1);
                if(ImGui::DragFloat("##YY", &yy, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(1, 1) = yy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix3d>) {
                float yy = arg(1, 1);
                if(ImGui::DragFloat("##YY", &yy, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(1, 1) = yy;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // YZ
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
        ImGui::Button("yz", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Matrix3i>) {
                int yz = arg(1, 2);
                if(ImGui::DragInt("##YZ", &yz, 0.1f, 0.0f, 0.0f)) {
                    arg(1, 2) = yz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix3f>) {
                float yz = arg(1, 2);
                if(ImGui::DragFloat("##YZ", &yz, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(1, 2) = yz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix3d>) {
                float yz = arg(1, 2);
                if(ImGui::DragFloat("##YZ", &yz, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(1, 2) = yz;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    // ZX
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0f, 1.0f, 1.0f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::Button("zx", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Matrix3i>) {
                int zx = arg(2, 0);
                if(ImGui::DragInt("##ZX", &zx, 0.1f, 0.0f, 0.0f)) {
                    arg(2, 0) = zx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix3f>) {
                float zx = arg(2, 0);
                if(ImGui::DragFloat("##ZX", &zx, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(2, 0) = zx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix3d>) {
                float zx = arg(2, 0);
                if(ImGui::DragFloat("##ZY", &zx, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(2, 0) = zx;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // ZY
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::Button("zy", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Matrix3i>) {
                int zy = arg(2, 1);
                if(ImGui::DragInt("##ZY", &zy, 0.1f, 0.0f, 0.0f)) {
                    arg(2, 1) = zy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix3f>) {
                float zy = arg(2, 1);
                if(ImGui::DragFloat("##ZY", &zy, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(2, 1) = zy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix3d>) {
                float zy = arg(2, 1);
                if(ImGui::DragFloat("##ZY", &zy, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(2, 1) = zy;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // ZZ
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
        ImGui::Button("zz", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Matrix3i>) {
                int zz = arg(2, 2);
                if(ImGui::DragInt("##ZZ", &zz, 0.1f, 0.0f, 0.0f)) {
                    arg(2, 2) = zz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix3f>) {
                float zz = arg(2, 2);
                if(ImGui::DragFloat("##ZZ", &zz, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(2, 2) = zz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Matrix3d>) {
                float zz = arg(2, 2);
                if(ImGui::DragFloat("##ZZ", &zz, 0.1f, 0.0f, 0.0f, "%.2f")) {
                    arg(2, 2) = zz;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::PopStyleVar();

}

//////////////////////////////////////////////////////////////////////////////

void EditorPropertyTransform2::OnImGuiContent() {
    if (type_category_ == TypeCategory::Isometry2f) {
        data_ = property_data_model_->GetValue().convert<Isometry2f>();
    } else if (type_category_ == TypeCategory::Isometry2d) {
        data_ = property_data_model_->GetValue().convert<Isometry2d>();
    } else if (type_category_ == TypeCategory::Affine2f) {
        data_ = property_data_model_->GetValue().convert<Affine2f>();
    } else if (type_category_ == TypeCategory::Affine2d) {
        data_ = property_data_model_->GetValue().convert<Affine2d>();
    } else if (type_category_ == TypeCategory::Projective2f) {
        data_ = property_data_model_->GetValue().convert<Projective2f>();
    } else if (type_category_ == TypeCategory::Projective2d) {
        data_ = property_data_model_->GetValue().convert<Projective2d>();
    }

    auto divided_width = ImGui::GetContentRegionAvail().x / 2.0f;

    float frame_height = ImGui::GetFrameHeight();
    ImVec2 button_size = { frame_height + 3.0f, frame_height };

    ImVec2 innerItemSpacing = ImGui::GetStyle().ItemInnerSpacing;
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, innerItemSpacing);

    // XX
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0f, 1.0f, 1.0f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::Button("xx", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Isometry2f>) {
                float xx = arg(0, 0);
                if(ImGui::DragFloat("##XX", &xx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 0) = xx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Isometry2d>) {
                float xx = arg(0, 0);
                if(ImGui::DragFloat("##XX", &xx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 0) = xx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine2f>) {
                float xx = arg(0, 0);
                if(ImGui::DragFloat("##XX", &xx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 0) = xx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine2d>) {
                float xx = arg(0, 0);
                if(ImGui::DragFloat("##XX", &xx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 0) = xx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective2f>) {
                float xx = arg(0, 0);
                if(ImGui::DragFloat("##XX", &xx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 0) = xx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective2d>) {
                float xx = arg(0, 0);
                if(ImGui::DragFloat("##XX", &xx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 0) = xx;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // XY
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::Button("xy", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Isometry2f>) {
                float xy = arg(0, 1);
                if(ImGui::DragFloat("##XY", &xy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 1) = xy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Isometry2d>) {
                float xy = arg(0, 1);
                if(ImGui::DragFloat("##XY", &xy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 1) = xy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine2f>) {
                float xy = arg(0, 1);
                if(ImGui::DragFloat("##XY", &xy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 1) = xy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine2d>) {
                float xy = arg(0, 1);
                if(ImGui::DragFloat("##XY", &xy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 1) = xy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective2f>) {
                float xy = arg(0, 1);
                if(ImGui::DragFloat("##XY", &xy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 1) = xy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective2d>) {
                float xy = arg(0, 1);
                if(ImGui::DragFloat("##XY", &xy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 1) = xy;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    // YX
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0f, 1.0f, 1.0f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::Button("yx", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Isometry2f>) {
                float yz = arg(1, 0);
                if(ImGui::DragFloat("##YX", &yz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 0) = yz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Isometry2d>) {
                float yz = arg(1, 0);
                if(ImGui::DragFloat("##YX", &yz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 0) = yz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine2f>) {
                float yz = arg(1, 0);
                if(ImGui::DragFloat("##XY", &yz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 0) = yz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine2d>) {
                float yz = arg(1, 0);
                if(ImGui::DragFloat("##XY", &yz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 0) = yz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective2f>) {
                float yz = arg(1, 0);
                if(ImGui::DragFloat("##XY", &yz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 0) = yz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective2d>) {
                float yz = arg(1, 0);
                if(ImGui::DragFloat("##XY", &yz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 0) = yz;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // YY
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::Button("yy", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Isometry2f>) {
                float yy = arg(1, 1);
                if(ImGui::DragFloat("##YY", &yy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 1) = yy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Isometry2d>) {
                float yy = arg(1, 1);
                if(ImGui::DragFloat("##YY", &yy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 1) = yy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine2f>) {
                float yy = arg(1, 1);
                if(ImGui::DragFloat("##YY", &yy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 1) = yy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine2d>) {
                float yy = arg(1, 1);
                if(ImGui::DragFloat("##YY", &yy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 1) = yy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective2f>) {
                float yy = arg(1, 1);
                if(ImGui::DragFloat("##YY", &yy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 1) = yy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective2d>) {
                float yy = arg(1, 1);
                if(ImGui::DragFloat("##YY", &yy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 1) = yy;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::PopStyleVar();

}

//////////////////////////////////////////////////////////////

void EditorPropertyVectorX::OnImGuiContent() {
    if (type_category_ == TypeCategory::VectorXi) {
        auto matrix = property_data_model_->GetValue().convert<VectorXi>();
        rows_ = matrix.size();
        data_ = matrix;
    } else if (type_category_ == TypeCategory::VectorXf) {
        auto matrix = property_data_model_->GetValue().convert<VectorXf>();
        rows_ = matrix.size();
        data_ = matrix;
    } else if (type_category_ == TypeCategory::VectorXd) {
        auto matrix = property_data_model_->GetValue().convert<VectorXd>();
        rows_ = matrix.size();
        data_ = matrix;
    }

    auto divided_width = ImGui::GetContentRegionAvail().x;

    float frame_height = ImGui::GetFrameHeight();
    ImVec2 button_size = { frame_height + 3.0f, frame_height };

    ImVec2 innerItemSpacing = ImGui::GetStyle().ItemInnerSpacing;
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, innerItemSpacing);

    for (int i = 0; i < rows_; i++ ) {
        {
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.59f, 0.71f, 0.26f, 1.0f });
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.5f, 0.9f, 0.5f, 1.0f });
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.6f, 0.7f, 0.5f, 1.0f });
            ImGui::Button(std::to_string(i).c_str(), button_size);
            ImGui::PopStyleColor(4);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(divided_width - button_size.x);

            std::visit([i, this](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, VectorXi>) {
                    int data = arg[i];
                    if(ImGui::DragInt(fmt::format("##{}", i).c_str(), &data, 0.01f, 0.0f, 0.0f)) {
                        arg[i] = data;
                        property_data_model_->SetValue(arg);
                    }
                } else if constexpr (std::is_same_v<T, VectorXf>) {
                    float data = arg[i];
                    if(ImGui::DragFloat(fmt::format("##{}", i).c_str(), &data, 0.01f, 0.0f, 0.0f, "%.2f")) {
                        arg[i] = data;
                        property_data_model_->SetValue(arg);
                    }
                } else if constexpr (std::is_same_v<T, VectorXd>) {
                    float data = arg[i];
                    if(ImGui::DragFloat(fmt::format("##{}", i).c_str(), &data, 0.01f, 0.0f, 0.0f, "%.2f")) {
                        arg[i] = data;
                        property_data_model_->SetValue(arg);
                    }
                }
            }, data_);
            ImGui::PopStyleVar();
        }
    }

    ImGui::PopStyleVar();
}

//////////////////////////////////////////////////////////////

void EditorPropertyMatrixX::OnImGuiContent() {
    if (type_category_ == TypeCategory::MatrixXi) {
        auto matrix = property_data_model_->GetValue().convert<MatrixXi>();
        rows_ = matrix.rows();
        columns_ = matrix.cols();
        data_ = matrix;
    } else if (type_category_ == TypeCategory::MatrixXf) {
        auto matrix = property_data_model_->GetValue().convert<MatrixXf>();
        rows_ = matrix.rows();
        columns_ = matrix.cols();
        data_ = matrix;
    } else if (type_category_ == TypeCategory::MatrixXd) {
        auto matrix = property_data_model_->GetValue().convert<MatrixXd>();
        rows_ = matrix.rows();
        columns_ = matrix.cols();
        data_ = matrix;
    }

    float column_0_width = 40.0f;
    auto table_width = (ImGui::GetContentRegionAvail().x - column_0_width) / columns_;

    ImVec2 innerItemSpacing = ImGui::GetStyle().ItemInnerSpacing;
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, innerItemSpacing);

    if (ImGui::BeginTable("table_matrix", columns_ + 1, ImGuiTableFlags_BordersV | ImGuiTableFlags_BordersOuterH | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoBordersInBody)) {
        ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_NoHide | ImGuiTableColumnFlags_WidthStretch, column_0_width, 0);
        for(int j = 0; j < columns_; j++) {
            ImGui::TableSetupColumn(std::to_string(j).c_str(), ImGuiTableColumnFlags_NoHide | ImGuiTableColumnFlags_WidthStretch, table_width, j+1);
        }
        ImGui::TableSetupScrollFreeze(0, 1); // Make row always visible
        ImGui::TableHeadersRow();

        for (int i = 0; i < rows_; i++) {
            ImGui::TableNextRow();
            for (int j = 0; j < columns_ + 1; j++) {
                ImGui::TableSetColumnIndex(j);
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (j == 0) {
                    ImGui::TextUnformatted(std::to_string(i).c_str());
                } else {
                    std::visit([i, j, this](auto&& arg) {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, MatrixXi>) {
                            int data = arg(i, j-1);
                            if(ImGui::DragInt(fmt::format("##{},{}", i, j-1).c_str(), &data, 0.01f, 0.0f, 0.0f)) {
                                arg(i, j-1) = data;
                                property_data_model_->SetValue(arg);
                            }
                        } else if constexpr (std::is_same_v<T, MatrixXf>) {
                            float data = arg(i, j-1);
                            if(ImGui::DragFloat(fmt::format("##{},{}", i, j-1).c_str(), &data, 0.01f, 0.0f, 0.0f, "%.2f")) {
                                arg(i, j-1) = data;
                                property_data_model_->SetValue(arg);
                            }
                        } else if constexpr (std::is_same_v<T, MatrixXd>) {
                            float data = arg(i, j-1);
                            if(ImGui::DragFloat(fmt::format("##{},{}", i, j-1).c_str(), &data, 0.01f, 0.0f, 0.0f, "%.2f")) {
                                arg(i, j-1) = data;
                                property_data_model_->SetValue(arg);
                            }
                        }
                    }, data_);
                }
            }
        }

        ImGui::EndTable();
    }

    ImGui::PopStyleVar();
}



///////////////////////////////////////////////////////////////////////////

void EditorPropertyTransform3::OnImGuiContent() {
    if (type_category_ == TypeCategory::Isometry3f) {
        data_ = property_data_model_->GetValue().convert<Isometry3f>();
    } else if (type_category_ == TypeCategory::Isometry3d) {
        data_ = property_data_model_->GetValue().convert<Isometry3d>();
    } else if (type_category_ == TypeCategory::Affine3f) {
        data_ = property_data_model_->GetValue().convert<Affine3f>();
    } else if (type_category_ == TypeCategory::Affine3d) {
        data_ = property_data_model_->GetValue().convert<Affine3d>();
    } else if (type_category_ == TypeCategory::Projective3f) {
        data_ = property_data_model_->GetValue().convert<Projective3f>();
    } else if (type_category_ == TypeCategory::Projective3d) {
        data_ = property_data_model_->GetValue().convert<Projective3d>();
    }

    auto divided_width = ImGui::GetContentRegionAvail().x / 3.0f;

    float frame_height = ImGui::GetFrameHeight();
    ImVec2 button_size = { frame_height + 3.0f, frame_height };

    ImVec2 innerItemSpacing = ImGui::GetStyle().ItemInnerSpacing;
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, innerItemSpacing);

    // XX
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0f, 1.0f, 1.0f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::Button("xx", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);
        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Isometry3f>) {
                float xx = arg(0, 0);
                if(ImGui::DragFloat("##XX", &xx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 0) = xx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Isometry3d>) {
                float xx = arg(0, 0);
                if(ImGui::DragFloat("##XX", &xx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 0) = xx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine3f>) {
                float xx = arg(0, 0);
                if(ImGui::DragFloat("##XX", &xx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 0) = xx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine3d>) {
                float xx = arg(0, 0);
                if(ImGui::DragFloat("##XX", &xx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 0) = xx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective3f>) {
                float xx = arg(0, 0);
                if(ImGui::DragFloat("##XX", &xx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 0) = xx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective3d>) {
                float xx = arg(0, 0);
                if(ImGui::DragFloat("##XX", &xx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 0) = xx;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // XY
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::Button("xy", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Isometry3f>) {
                float xy = arg(0, 1);
                if(ImGui::DragFloat("##XY", &xy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 1) = xy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Isometry3d>) {
                float xy = arg(0, 1);
                if(ImGui::DragFloat("##XY", &xy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 1) = xy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine3f>) {
                float xy = arg(0, 1);
                if(ImGui::DragFloat("##XY", &xy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 1) = xy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine3d>) {
                float xy = arg(0, 1);
                if(ImGui::DragFloat("##XY", &xy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 1) = xy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective3f>) {
                float xy = arg(0, 1);
                if(ImGui::DragFloat("##XY", &xy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 1) = xy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective3d>) {
                float xy = arg(0, 1);
                if(ImGui::DragFloat("##XY", &xy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 1) = xy;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // XZ
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
        ImGui::Button("xz", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Isometry3f>) {
                float xz = arg(0, 2);
                if(ImGui::DragFloat("##XZ", &xz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 2) = xz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Isometry3d>) {
                float xz = arg(0, 2);
                if(ImGui::DragFloat("##XZ", &xz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 2) = xz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine3f>) {
                float xz = arg(0, 2);
                if(ImGui::DragFloat("##XZ", &xz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 2) = xz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine3d>) {
                float xz = arg(0, 2);
                if(ImGui::DragFloat("##XZ", &xz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 2) = xz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective3f>) {
                float xz = arg(0, 2);
                if(ImGui::DragFloat("##XZ", &xz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 2) = xz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective3d>) {
                float xz = arg(0, 1);
                if(ImGui::DragFloat("##XZ", &xz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(0, 1) = xz;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    // YX
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0f, 1.0f, 1.0f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::Button("yx", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Isometry3f>) {
                float yx = arg(1, 0);
                if(ImGui::DragFloat("##YX", &yx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 0) = yx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Isometry3d>) {
                float yx = arg(1, 0);
                if(ImGui::DragFloat("##YX", &yx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 0) = yx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine3f>) {
                float yx = arg(1, 0);
                if(ImGui::DragFloat("##YX", &yx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 0) = yx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine3d>) {
                float yx = arg(1, 0);
                if(ImGui::DragFloat("##YX", &yx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 0) = yx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective3f>) {
                float yx = arg(1, 0);
                if(ImGui::DragFloat("##YX", &yx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 0) = yx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective3d>) {
                float yx = arg(1, 0);
                if(ImGui::DragFloat("##YX", &yx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 0) = yx;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // YY
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::Button("yy", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Isometry3f>) {
                float yy = arg(1, 1);
                if(ImGui::DragFloat("##YY", &yy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 1) = yy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Isometry3d>) {
                float yy = arg(1, 1);
                if(ImGui::DragFloat("##YY", &yy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 0) = yy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine3f>) {
                float yy = arg(1, 1);
                if(ImGui::DragFloat("##YX", &yy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 1) = yy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine3d>) {
                float yy = arg(1, 1);
                if(ImGui::DragFloat("##YX", &yy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 1) = yy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective3f>) {
                float yy = arg(1, 1);
                if(ImGui::DragFloat("##YX", &yy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 1) = yy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective3d>) {
                float yy = arg(1, 1);
                if(ImGui::DragFloat("##YX", &yy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 1) = yy;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // YZ
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
        ImGui::Button("yz", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Isometry3f>) {
                float yz = arg(1, 2);
                if(ImGui::DragFloat("##YZ", &yz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 2) = yz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Isometry3d>) {
                float yz = arg(1, 2);
                if(ImGui::DragFloat("##YZ", &yz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 2) = yz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine3f>) {
                float yz = arg(1, 2);
                if(ImGui::DragFloat("##YZ", &yz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 2) = yz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine3d>) {
                float yz = arg(1, 2);
                if(ImGui::DragFloat("##YZ", &yz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 2) = yz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective3f>) {
                float yz = arg(1, 2);
                if(ImGui::DragFloat("##YZ", &yz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 2) = yz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective3d>) {
                float yz = arg(1, 2);
                if(ImGui::DragFloat("##YZ", &yz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(1, 2) = yz;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    // ZX
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0f, 1.0f, 1.0f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::Button("zx", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Isometry3f>) {
                float zx = arg(2, 0);
                if(ImGui::DragFloat("##ZX", &zx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(2, 0) = zx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Isometry3d>) {
                float zx = arg(2, 0);
                if(ImGui::DragFloat("##ZX", &zx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(2, 0) = zx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine3f>) {
                float zx = arg(2, 0);
                if(ImGui::DragFloat("##ZX", &zx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(2, 0) = zx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine3d>) {
                float zx = arg(2, 0);
                if(ImGui::DragFloat("##ZX", &zx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(2, 0) = zx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective3f>) {
                float zx = arg(2, 0);
                if(ImGui::DragFloat("##ZX", &zx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(2, 0) = zx;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective3d>) {
                float zx = arg(2, 0);
                if(ImGui::DragFloat("##ZX", &zx, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(2, 0) = zx;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // ZY
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
        ImGui::Button("zy", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Isometry3f>) {
                float zy = arg(2, 1);
                if(ImGui::DragFloat("##ZY", &zy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(2, 1) = zy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Isometry3d>) {
                float zy = arg(2, 1);
                if(ImGui::DragFloat("##ZY", &zy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(2, 1) = zy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine3f>) {
                float zy = arg(2, 1);
                if(ImGui::DragFloat("##ZY", &zy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(2, 1) = zy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine3d>) {
                float zy = arg(2, 1);
                if(ImGui::DragFloat("##ZY", &zy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(2, 1) = zy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective3f>) {
                float zy = arg(2, 1);
                if(ImGui::DragFloat("##ZY", &zy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(2, 1) = zy;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective3d>) {
                float zy = arg(2, 1);
                if(ImGui::DragFloat("##ZY", &zy, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(2, 1) = zy;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // ZZ
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
        ImGui::Button("zz", button_size);
        ImGui::PopStyleColor(4);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(divided_width - button_size.x);

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Isometry3f>) {
                float zz = arg(2, 2);
                if(ImGui::DragFloat("##ZZ", &zz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(2, 2) = zz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Isometry3d>) {
                float zz = arg(2, 2);
                if(ImGui::DragFloat("##ZZ", &zz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(2, 2) = zz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine3f>) {
                float zz = arg(2, 2);
                if(ImGui::DragFloat("##ZZ", &zz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(2, 2) = zz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Affine3d>) {
                float zz = arg(2, 2);
                if(ImGui::DragFloat("##ZZ", &zz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(2, 2) = zz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective3f>) {
                float zz = arg(2, 2);
                if(ImGui::DragFloat("##ZZ", &zz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(2, 2) = zz;
                    property_data_model_->SetValue(arg);
                }
            } else if constexpr (std::is_same_v<T, Projective3d>) {
                float zz = arg(2, 2);
                if(ImGui::DragFloat("##ZZ", &zz, 0.01f, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                    arg(2, 2) = zz;
                    property_data_model_->SetValue(arg);
                }
            }
        }, data_);
        ImGui::PopStyleVar();
    }

    ImGui::PopStyleVar();
}

}
