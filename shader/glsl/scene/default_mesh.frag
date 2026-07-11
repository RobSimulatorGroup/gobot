#version 450 core

in vec3 v_world_position;
in vec3 v_world_normal;
in vec4 v_color;

out vec4 frag_color;

uniform vec4 u_color;
uniform vec3 u_camera_position;
uniform float u_metallic;
uniform float u_roughness;
uniform float u_specular;

void main() {
    vec3 normal = normalize(v_world_normal);
    vec3 geometric_normal = normalize(cross(dFdx(v_world_position), dFdy(v_world_position)));
    if (dot(geometric_normal, normal) < 0.0) {
        geometric_normal = -geometric_normal;
    }
    normal = normalize(mix(normal, geometric_normal, 0.25));
    vec3 light_dir = normalize(vec3(0.35, 0.45, 0.82));
    vec3 view_dir = normalize(u_camera_position - v_world_position);
    vec3 half_dir = normalize(light_dir + view_dir);
    float roughness = clamp(u_roughness, 0.04, 1.0);
    float metallic = clamp(u_metallic, 0.0, 1.0);
    float specular_weight = clamp(u_specular, 0.0, 1.0);
    vec4 base_color = u_color * v_color;

    float n_dot_l = max(dot(normal, light_dir), 0.0);
    float n_dot_h = max(dot(normal, half_dir), 0.0);
    float v_dot_h = max(dot(view_dir, half_dir), 0.0);
    float shininess = mix(128.0, 4.0, roughness * roughness);
    vec3 dielectric_f0 = mix(vec3(0.02), vec3(0.08), specular_weight);
    vec3 f0 = mix(dielectric_f0, base_color.rgb, metallic);
    vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - v_dot_h, 5.0);
    float specular = pow(n_dot_h, shininess) * (1.0 - roughness * 0.75);

    vec3 diffuse_color = base_color.rgb * (1.0 - metallic);
    vec3 sky_color = vec3(0.55, 0.62, 0.70);
    vec3 ground_color = vec3(0.08, 0.085, 0.09);
    float sky_mix = clamp(normal.z * 0.5 + 0.5, 0.0, 1.0);
    vec3 environment_color = mix(ground_color, sky_color, sky_mix);
    vec3 ambient = diffuse_color * (vec3(0.28) + environment_color * 0.22);
    vec3 direct = diffuse_color * n_dot_l * 0.78;
    vec3 environment_specular =
            environment_color * fresnel * mix(0.12, 0.45, metallic) * (1.0 - roughness * 0.65);
    vec3 lit = ambient + direct + fresnel * specular * 0.35 + environment_specular;
    vec3 mapped = vec3(1.0) - exp(-lit * 1.15);
    vec3 display_color = pow(clamp(mapped, 0.0, 1.0), vec3(1.0 / 2.2));
    frag_color = vec4(display_color, base_color.a);
}
