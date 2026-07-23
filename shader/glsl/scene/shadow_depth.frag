#version 450 core

in vec2 v_uv;
in float v_vertex_alpha;

uniform float u_alpha;
uniform float u_alpha_cutoff;
uniform bool u_has_albedo_texture;
uniform sampler2D u_albedo_texture;

void main() {
    float alpha = u_alpha * v_vertex_alpha;
    if (u_has_albedo_texture) {
        alpha *= texture(u_albedo_texture, v_uv).a;
    }
    if (alpha < u_alpha_cutoff) {
        discard;
    }
}
