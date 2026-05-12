#version 460 core

in vec3 v_world_position;
in vec3 v_world_normal;

out vec4 frag_color;

uniform vec4 u_color;
uniform vec3 u_camera_position;
uniform float u_metallic;
uniform float u_roughness;
uniform float u_specular;

void main() {
    vec3 normal = normalize(v_world_normal);
    vec3 light_dir = normalize(vec3(0.35, 0.45, 0.82));
    vec3 view_dir = normalize(u_camera_position - v_world_position);
    vec3 half_dir = normalize(light_dir + view_dir);
    float roughness = clamp(u_roughness, 0.04, 1.0);
    float metallic = clamp(u_metallic, 0.0, 1.0);
    float specular_weight = clamp(u_specular, 0.0, 1.0);

    float n_dot_l = max(dot(normal, light_dir), 0.0);
    float n_dot_h = max(dot(normal, half_dir), 0.0);
    float v_dot_h = max(dot(view_dir, half_dir), 0.0);
    float shininess = mix(128.0, 4.0, roughness * roughness);
    vec3 dielectric_f0 = mix(vec3(0.02), vec3(0.08), specular_weight);
    vec3 f0 = mix(dielectric_f0, u_color.rgb, metallic);
    vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - v_dot_h, 5.0);
    float specular = pow(n_dot_h, shininess) * (1.0 - roughness * 0.75);

    vec3 diffuse_color = u_color.rgb * (1.0 - metallic);
    vec3 ambient_diffuse = diffuse_color * 0.18;

    vec3 sky_color = vec3(0.55, 0.62, 0.70);
    vec3 ground_color = vec3(0.08, 0.085, 0.09);
    float sky_mix = normal.z * 0.5 + 0.5;
    vec3 environment_color = mix(ground_color, sky_color, sky_mix);
    vec3 environment_specular = environment_color * fresnel * mix(0.20, 0.85, metallic) * (1.0 - roughness * 0.65);

    vec3 lit = ambient_diffuse + diffuse_color * n_dot_l * 0.82 + fresnel * specular + environment_specular;
    frag_color = vec4(lit, u_color.a);
}
