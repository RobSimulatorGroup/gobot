#pragma once

#include "gobot/core/rid.hpp"

#include "gobot_export.h"

struct ImDrawList;
struct ImVec2;

namespace gobot {

class Camera3D;
class Node;

class GOBOT_EXPORT EditorViewportRenderer {
public:
    void Render(const RID& viewport, const Node* scene_root, const Camera3D* camera);

    Node* PickNode(Node* scene_root,
                   const Camera3D* camera,
                   const ImVec2& viewport_position,
                   const ImVec2& viewport_size,
                   const ImVec2& mouse_position) const;

    void RenderOverlay(const Node* scene_root,
                       const Camera3D* camera,
                       const ImVec2& viewport_position,
                       const ImVec2& viewport_size,
                       ImDrawList* draw_list,
                       const Node* hovered_node,
                       const Node* motion_target_node);
};

}
