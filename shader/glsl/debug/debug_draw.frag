#version 450 core

layout(location = 0) in vec3 v_world_position;

out vec4 frag_color;

uniform vec4 u_color;
uniform bool u_surface_shading;

void main() {
    vec3 color = u_color.rgb;
    if (u_surface_shading) {
        vec3 normal = normalize(cross(dFdx(v_world_position), dFdy(v_world_position)));
        vec3 light_direction = normalize(vec3(-0.35, 0.45, 0.82));
        float diffuse = 0.38 + 0.62 * abs(dot(normal, light_direction));
        color *= diffuse;
    }
    frag_color = vec4(color, u_color.a);
}
