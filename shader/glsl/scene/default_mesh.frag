#version 450 core

in vec3 v_world_position;
in vec3 v_world_normal;
in vec4 v_color;
in vec4 v_world_tangent;
in vec2 v_uv;

out vec4 frag_color;

uniform vec4 u_color;
uniform vec3 u_camera_position;
uniform float u_metallic;
uniform float u_roughness;
uniform float u_specular;
uniform vec3 u_light_direction;
uniform vec3 u_light_color;
uniform float u_light_intensity;
uniform vec3 u_sky_color;
uniform vec3 u_ground_color;
uniform float u_ambient_intensity;
uniform float u_exposure;
uniform float u_normal_scale;
uniform float u_occlusion_strength;
uniform vec3 u_emissive;
uniform int u_alpha_mode;
uniform float u_alpha_cutoff;
uniform bool u_has_albedo_texture;
uniform bool u_has_metallic_roughness_texture;
uniform bool u_has_normal_texture;
uniform bool u_has_occlusion_texture;
uniform bool u_has_emissive_texture;
uniform bool u_has_environment_texture;
uniform sampler2D u_albedo_texture;
uniform sampler2D u_metallic_roughness_texture;
uniform sampler2D u_normal_texture;
uniform sampler2D u_occlusion_texture;
uniform sampler2D u_emissive_texture;
uniform sampler2D u_environment_texture;
uniform mat3 u_environment_rotation;
uniform float u_environment_intensity;

const int MAX_LIGHTS = 16;
uniform int u_light_count;
uniform vec4 u_light_position_type[MAX_LIGHTS];
uniform vec4 u_light_direction_range[MAX_LIGHTS];
uniform vec4 u_light_color_intensity[MAX_LIGHTS];
uniform vec2 u_light_spot_cosines[MAX_LIGHTS];

const float PI = 3.14159265359;

float distribution_ggx(vec3 normal, vec3 half_dir, float roughness) {
    float alpha = roughness * roughness;
    float alpha_squared = alpha * alpha;
    float n_dot_h = max(dot(normal, half_dir), 0.0);
    float denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
    return alpha_squared / max(PI * denominator * denominator, 0.0001);
}

float geometry_schlick_ggx(float n_dot_direction, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return n_dot_direction / max(n_dot_direction * (1.0 - k) + k, 0.0001);
}

vec3 fresnel_schlick(float cosine, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosine, 0.0, 1.0), 5.0);
}

vec3 fresnel_schlick_roughness(float cosine, vec3 f0, float roughness) {
    return f0 + (max(vec3(1.0 - roughness), f0) - f0) *
            pow(clamp(1.0 - cosine, 0.0, 1.0), 5.0);
}

vec3 aces_film(vec3 color) {
    return clamp((color * (2.51 * color + 0.03)) /
                 (color * (2.43 * color + 0.59) + 0.14),
                 0.0,
                 1.0);
}

vec2 direction_to_equirect_uv(vec3 direction) {
    direction = normalize(u_environment_rotation * direction);
    return vec2(atan(direction.y, direction.x) / (2.0 * PI) + 0.5,
                acos(clamp(direction.z, -1.0, 1.0)) / PI);
}

vec3 evaluate_direct_light(vec3 normal,
                           vec3 view_dir,
                           vec3 light_dir,
                           vec3 radiance,
                           vec3 base_color,
                           float metallic,
                           float roughness,
                           float specular_weight) {
    vec3 half_dir = normalize(light_dir + view_dir);
    float n_dot_l = max(dot(normal, light_dir), 0.0);
    float n_dot_v = max(dot(normal, view_dir), 0.0);
    float v_dot_h = max(dot(view_dir, half_dir), 0.0);
    vec3 dielectric_f0 = mix(vec3(0.02), vec3(0.08), specular_weight);
    vec3 f0 = mix(dielectric_f0, base_color, metallic);
    vec3 fresnel = fresnel_schlick(v_dot_h, f0);
    float distribution = distribution_ggx(normal, half_dir, roughness);
    float geometry = geometry_schlick_ggx(n_dot_v, roughness) *
                     geometry_schlick_ggx(n_dot_l, roughness);
    vec3 specular = distribution * geometry * fresnel /
                    max(4.0 * n_dot_v * n_dot_l, 0.0001);
    vec3 diffuse_weight = (1.0 - fresnel) * (1.0 - metallic);
    return (diffuse_weight * base_color / PI + specular) * radiance * n_dot_l;
}

void main() {
    vec3 normal = normalize(v_world_normal);
    if (!gl_FrontFacing) {
        normal = -normal;
    }
    if (u_has_normal_texture) {
        vec3 tangent = normalize(v_world_tangent.xyz - normal * dot(normal, v_world_tangent.xyz));
        vec3 bitangent = normalize(cross(normal, tangent)) * v_world_tangent.w;
        vec3 tangent_normal = texture(u_normal_texture, v_uv).xyz * 2.0 - 1.0;
        tangent_normal.xy *= u_normal_scale;
        normal = normalize(mat3(tangent, bitangent, normal) * normalize(tangent_normal));
    }
    vec3 geometric_normal = normalize(cross(dFdx(v_world_position), dFdy(v_world_position)));
    if (dot(geometric_normal, normal) < 0.0) {
        geometric_normal = -geometric_normal;
    }
    normal = normalize(mix(normal, geometric_normal, 0.18));
    vec3 view_dir = normalize(u_camera_position - v_world_position);
    vec4 sampled_albedo = vec4(1.0);
    if (u_has_albedo_texture) {
        sampled_albedo = texture(u_albedo_texture, v_uv);
        sampled_albedo.rgb = pow(max(sampled_albedo.rgb, vec3(0.0)), vec3(2.2));
    }
    vec4 base_color = u_color * v_color * sampled_albedo;
    if (u_alpha_mode == 1 && base_color.a < u_alpha_cutoff) {
        discard;
    }

    vec4 metallic_roughness = u_has_metallic_roughness_texture
            ? texture(u_metallic_roughness_texture, v_uv)
            : vec4(1.0);
    float roughness = clamp(u_roughness * metallic_roughness.g, 0.04, 1.0);
    float metallic = clamp(u_metallic * metallic_roughness.b, 0.0, 1.0);
    float specular_weight = clamp(u_specular, 0.0, 1.0);

    float n_dot_v = max(dot(normal, view_dir), 0.0);
    vec3 dielectric_f0 = mix(vec3(0.02), vec3(0.08), specular_weight);
    vec3 f0 = mix(dielectric_f0, base_color.rgb, metallic);
    vec3 direct = vec3(0.0);
    for (int i = 0; i < u_light_count; ++i) {
        int light_type = int(u_light_position_type[i].w + 0.5);
        vec3 light_dir;
        float attenuation = 1.0;
        if (light_type == 0) {
            light_dir = normalize(u_light_direction_range[i].xyz);
        } else {
            vec3 to_light = u_light_position_type[i].xyz - v_world_position;
            float distance_to_light = length(to_light);
            light_dir = to_light / max(distance_to_light, 0.0001);
            float range = max(u_light_direction_range[i].w, 0.001);
            float range_weight = clamp(1.0 - distance_to_light / range, 0.0, 1.0);
            attenuation = range_weight * range_weight / max(distance_to_light * distance_to_light, 0.01);
            if (light_type == 2) {
                float cone_cosine = dot(-light_dir, normalize(u_light_direction_range[i].xyz));
                attenuation *= smoothstep(u_light_spot_cosines[i].y,
                                          u_light_spot_cosines[i].x,
                                          cone_cosine);
            }
        }
        vec3 radiance = u_light_color_intensity[i].rgb *
                        u_light_color_intensity[i].a * attenuation;
        direct += evaluate_direct_light(normal,
                                        view_dir,
                                        light_dir,
                                        radiance,
                                        base_color.rgb,
                                        metallic,
                                        roughness,
                                        specular_weight);
    }

    float sky_mix = clamp(normal.z * 0.5 + 0.5, 0.0, 1.0);
    vec3 environment_color = mix(u_ground_color, u_sky_color, sky_mix);
    vec3 environment_specular_color = environment_color;
    if (u_has_environment_texture) {
        environment_color = texture(u_environment_texture,
                                    direction_to_equirect_uv(normal)).rgb * u_environment_intensity;
        vec3 reflection = reflect(-view_dir, normal);
        environment_specular_color = texture(u_environment_texture,
                                             direction_to_equirect_uv(reflection)).rgb *
                                     u_environment_intensity;
    }
    vec3 environment_fresnel = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    vec3 ambient_diffuse = (1.0 - environment_fresnel) *
                           (1.0 - metallic) * base_color.rgb;
    vec3 ambient_specular = environment_fresnel * mix(0.08, 0.32, 1.0 - roughness);
    float occlusion = u_has_occlusion_texture
            ? mix(1.0, texture(u_occlusion_texture, v_uv).r, u_occlusion_strength)
            : 1.0;
    vec3 ambient = (ambient_diffuse * environment_color +
                    ambient_specular * environment_specular_color) * u_ambient_intensity;
    ambient *= occlusion;

    vec3 emissive = u_emissive;
    if (u_has_emissive_texture) {
        vec3 sampled_emissive = texture(u_emissive_texture, v_uv).rgb;
        emissive *= pow(max(sampled_emissive, vec3(0.0)), vec3(2.2));
    }

    vec3 mapped = aces_film((ambient + direct + emissive) * u_exposure);
    vec3 display_color = pow(clamp(mapped, 0.0, 1.0), vec3(1.0 / 2.2));
    frag_color = vec4(display_color, base_color.a);
}
