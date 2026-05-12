#version 460 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_world_position;
out vec3 v_world_normal;

void main() {
    vec4 world_position = u_model * vec4(a_position, 1.0);
    v_world_position = world_position.xyz;
    v_world_normal = normalize(transpose(inverse(mat3(u_model))) * a_normal);
    gl_Position = u_projection * u_view * world_position;
}
