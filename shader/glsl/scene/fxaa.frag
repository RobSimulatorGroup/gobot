#version 450 core

in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

uniform sampler2D u_color_texture;
uniform vec2 u_inverse_size;

float luma(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

void main() {
    vec3 center = texture(u_color_texture, v_uv).rgb;
    float luma_center = luma(center);
    float luma_north = luma(textureOffset(u_color_texture, v_uv, ivec2(0, 1)).rgb);
    float luma_south = luma(textureOffset(u_color_texture, v_uv, ivec2(0, -1)).rgb);
    float luma_east = luma(textureOffset(u_color_texture, v_uv, ivec2(1, 0)).rgb);
    float luma_west = luma(textureOffset(u_color_texture, v_uv, ivec2(-1, 0)).rgb);

    float luma_min = min(luma_center, min(min(luma_north, luma_south), min(luma_east, luma_west)));
    float luma_max = max(luma_center, max(max(luma_north, luma_south), max(luma_east, luma_west)));
    float contrast = luma_max - luma_min;
    if (contrast < max(0.0312, luma_max * 0.125)) {
        frag_color = vec4(center, 1.0);
        return;
    }

    vec2 direction = vec2(-(luma_north - luma_south), luma_east - luma_west);
    float direction_reduce = max((luma_north + luma_south + luma_east + luma_west) * 0.03125,
                                 1.0 / 128.0);
    float reciprocal = 1.0 / (min(abs(direction.x), abs(direction.y)) + direction_reduce);
    direction = clamp(direction * reciprocal, vec2(-8.0), vec2(8.0)) * u_inverse_size;

    vec3 sample_a = 0.5 * (
            texture(u_color_texture, v_uv + direction * (1.0 / 3.0 - 0.5)).rgb +
            texture(u_color_texture, v_uv + direction * (2.0 / 3.0 - 0.5)).rgb);
    vec3 sample_b = sample_a * 0.5 + 0.25 * (
            texture(u_color_texture, v_uv + direction * -0.5).rgb +
            texture(u_color_texture, v_uv + direction * 0.5).rgb);
    float luma_b = luma(sample_b);
    frag_color = vec4(luma_b < luma_min || luma_b > luma_max ? sample_a : sample_b, 1.0);
}
