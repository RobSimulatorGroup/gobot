#version 450 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec4 a_color;
layout(location = 3) in vec4 a_tangent;
layout(location = 4) in vec2 a_uv;

uniform mat4 u_model;
uniform mat3 u_normal_matrix;
uniform mat4 u_view_projection;

out vec3 v_world_position;
out vec3 v_world_normal;
out vec4 v_color;
out vec4 v_world_tangent;
out vec2 v_uv;

void main() {
    vec4 world_position = u_model * vec4(a_position, 1.0);
    v_world_position = world_position.xyz;
    v_world_normal = normalize(u_normal_matrix * a_normal);
    v_color = a_color;
    v_world_tangent = vec4(normalize(mat3(u_model) * a_tangent.xyz), a_tangent.w);
    v_uv = a_uv;
    gl_Position = u_view_projection * world_position;
}
