#version 450 core

layout(location = 0) in vec3 a_position;
layout(location = 2) in vec4 a_color;
layout(location = 4) in vec2 a_uv;

uniform mat4 u_model;
uniform mat4 u_light_view_projection;

out vec2 v_uv;
out float v_vertex_alpha;

void main() {
    v_uv = a_uv;
    v_vertex_alpha = a_color.a;
    gl_Position = u_light_view_projection * u_model * vec4(a_position, 1.0);
}
