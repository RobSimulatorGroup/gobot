#include "luisa_renderer_internal.hpp"

namespace gobot::luisa_renderer {

void LuisaRenderer::CompileShaders() {
        Callable tea = [](UInt value0, UInt value1) noexcept {
            UInt sum = def(0u);
            $for (i, 4u) {
                sum += 0x9e3779b9u;
                value0 += ((value1 << 4u) + 0xa341316cu) ^ (value1 + sum) ^
                          ((value1 >> 5u) + 0xc8013ea4u);
                value1 += ((value0 << 4u) + 0xad90777du) ^ (value0 + sum) ^
                          ((value0 >> 5u) + 0x7e95761eu);
            };
            return value0;
        };
        Callable random_float = [](UInt& state) noexcept {
            state = state * 1664525u + 1013904223u;
            return cast<float>(state & 0x00ffffffu) * (1.0f / 16777216.0f);
        };
        Callable make_onb = [](Float3 normal) noexcept {
            Float3 binormal = normalize(ite(abs(normal.x) > abs(normal.z),
                                            make_float3(-normal.y, normal.x, 0.0f),
                                            make_float3(0.0f, -normal.z, normal.y)));
            return def<Onb>(normalize(cross(binormal, normal)), binormal, normal);
        };
        Callable cosine_sample = [](Float2 sample) noexcept {
            Float radius = sqrt(sample.x);
            Float phi = 2.0f * constants::pi * sample.y;
            return make_float3(radius * cos(phi), radius * sin(phi), sqrt(1.0f - sample.x));
        };
        Callable fresnel = [](Float cosine, Float3 f0) noexcept {
            return f0 + (1.0f - f0) * pow(clamp(1.0f - cosine, 0.0f, 1.0f), 5.0f);
        };
        Callable direct_brdf = [&](Float3 normal,
                                   Float3 view,
                                   Float3 light,
                                   Float3 radiance,
                                   Float3 albedo,
                                   Float metallic,
                                   Float roughness,
                                   Float specular_weight) noexcept {
            Float3 half_vector = normalize(view + light);
            Float n_dot_l = max(dot(normal, light), 0.0f);
            Float n_dot_v = max(dot(normal, view), 0.0f);
            Float n_dot_h = max(dot(normal, half_vector), 0.0f);
            Float v_dot_h = max(dot(view, half_vector), 0.0f);
            Float alpha = roughness * roughness;
            Float alpha2 = alpha * alpha;
            Float denominator = n_dot_h * n_dot_h * (alpha2 - 1.0f) + 1.0f;
            Float distribution = alpha2 / max(constants::pi * denominator * denominator, 1.0e-4f);
            Float k = (roughness + 1.0f) * (roughness + 1.0f) * 0.125f;
            Float geometry_v = n_dot_v / max(n_dot_v * (1.0f - k) + k, 1.0e-4f);
            Float geometry_l = n_dot_l / max(n_dot_l * (1.0f - k) + k, 1.0e-4f);
            Float3 f0 = lerp(make_float3(0.02f + 0.06f * specular_weight), albedo, metallic);
            Float3 f = fresnel(v_dot_h, f0);
            Float3 specular = distribution * geometry_v * geometry_l * f /
                              max(4.0f * n_dot_v * n_dot_l, 1.0e-4f);
            Float3 diffuse = (1.0f - f) * (1.0f - metallic) * albedo * constants::inv_pi;
            return (diffuse + specular) * radiance * n_dot_l;
        };

        Kernel2D clear_kernel = [](ImageFloat accumulation, ImageFloat guide) noexcept {
            UInt2 pixel = dispatch_id().xy();
            accumulation.write(pixel, make_float4(0.0f));
            guide.write(pixel, make_float4(0.0f));
        };
        Kernel2D seed_kernel = [tea](ImageUInt seeds, UInt frame_seed) noexcept {
            UInt2 pixel = dispatch_id().xy();
            seeds.write(pixel, make_uint4(tea(pixel.x ^ frame_seed, pixel.y), 0u, 0u, 0u));
        };

        Kernel2D trace_kernel = [=](ImageFloat accumulation,
                                    ImageUInt seeds,
                                    ImageFloat guide,
                                    AccelVar accel,
                                    BindlessVar geometry,
                                    BindlessVar textures,
                                    BufferVar<GpuMaterial> materials,
                                    BufferVar<GpuLight> lights,
                                    UInt light_count,
                                    UInt environment_texture,
                                    Float3 sky_color,
                                    Float3 ground_color,
                                    Float environment_intensity,
                                    Float4x4 inverse_view_projection,
                                    Float3 camera_position,
                                    UInt max_bounces,
                                    UInt frame_index) noexcept {
            set_block_size(16u, 16u, 1u);
            UInt2 pixel = dispatch_id().xy();
            Float2 resolution = make_float2(dispatch_size().xy());
            UInt state = seeds.read(pixel).x ^ tea(frame_index, pixel.x + pixel.y * dispatch_size().x);
            Float2 jitter = make_float2(random_float(state), random_float(state));
            Float2 ndc = (make_float2(pixel) + jitter) / resolution * 2.0f - 1.0f;
            Float4 near_h = inverse_view_projection * make_float4(ndc, -1.0f, 1.0f);
            Float4 far_h = inverse_view_projection * make_float4(ndc, 1.0f, 1.0f);
            Float3 near_point = near_h.xyz() / near_h.w;
            Float3 far_point = far_h.xyz() / far_h.w;
            Var<Ray> ray = make_ray(camera_position, normalize(far_point - near_point));
            Float3 throughput = def(make_float3(1.0f));
            Float3 radiance = def(make_float3(0.0f));
            Bool wrote_guide = def(false);

            $for (depth, 12u) {
                $if (depth >= max_bounces) { $break; };
                Var<TriangleHit> hit = accel.intersect(
                        ray, {.visibility_mask = kRgbVisibility});
                $if (hit->miss()) {
                    Float3 direction = ray->direction();
                    Float sky_mix = clamp(direction.z * 0.5f + 0.5f, 0.0f, 1.0f);
                    Float3 environment = lerp(ground_color, sky_color, sky_mix);
                    $if (environment_texture != kInvalidTexture) {
                        Float2 uv = make_float2(
                                atan2(direction.y, direction.x) / (2.0f * constants::pi) + 0.5f,
                                acos(clamp(direction.z, -1.0f, 1.0f)) / constants::pi);
                        environment = textures.tex2d(environment_texture).sample(uv).xyz() *
                                      environment_intensity;
                    };
                    radiance += throughput * environment;
                    $break;
                };

                Var<Triangle> triangle = geometry.buffer<Triangle>(hit.inst * 2u + 1u).read(hit.prim);
                Var<GpuVertex> vertex0 = geometry.buffer<GpuVertex>(hit.inst * 2u).read(triangle.i0);
                Var<GpuVertex> vertex1 = geometry.buffer<GpuVertex>(hit.inst * 2u).read(triangle.i1);
                Var<GpuVertex> vertex2 = geometry.buffer<GpuVertex>(hit.inst * 2u).read(triangle.i2);
                Float4x4 model = accel.instance_transform(hit.inst);
                Float4x4 normal_transform = transpose(inverse(model));
                Float3 local_position = triangle_interpolate(
                        hit.bary, vertex0.position, vertex1.position, vertex2.position);
                Float3 position = (model * make_float4(local_position, 1.0f)).xyz();
                Float3 normal = normalize((normal_transform * make_float4(
                        triangle_interpolate(hit.bary, vertex0.normal, vertex1.normal, vertex2.normal),
                        0.0f)).xyz());
                normal = faceforward(normal, ray->direction(), normal);
                Float4 tangent4 = triangle_interpolate(
                        hit.bary, vertex0.tangent, vertex1.tangent, vertex2.tangent);
                Float3 tangent = normalize((model * make_float4(tangent4.xyz(), 0.0f)).xyz());
                tangent = normalize(tangent - normal * dot(normal, tangent));
                Float3 bitangent = normalize(cross(normal, tangent)) * tangent4.w;
                Float2 uv = triangle_interpolate(hit.bary, vertex0.uv, vertex1.uv, vertex2.uv);
                Float4 vertex_color = triangle_interpolate(
                        hit.bary, vertex0.color, vertex1.color, vertex2.color);
                Var<GpuMaterial> material = materials.read(hit.inst);
                Float4 albedo = material.albedo * vertex_color;
                $if (material.textures0.x != kInvalidTexture) {
                    Float4 sampled = textures.tex2d(material.textures0.x).sample(uv);
                    albedo *= make_float4(pow(max(sampled.xyz(), make_float3(0.0f)), 2.2f), sampled.w);
                };
                Float metallic = material.pbr.x;
                Float roughness = clamp(material.pbr.y, 0.04f, 1.0f);
                $if (material.textures0.y != kInvalidTexture) {
                    Float4 sampled = textures.tex2d(material.textures0.y).sample(uv);
                    roughness = clamp(roughness * sampled.y, 0.04f, 1.0f);
                    metallic = clamp(metallic * sampled.z, 0.0f, 1.0f);
                };
                $if (material.textures0.z != kInvalidTexture) {
                    Float3 mapped = textures.tex2d(material.textures0.z).sample(uv).xyz() * 2.0f - 1.0f;
                    mapped = make_float3(mapped.xy() * material.pbr.w, mapped.z);
                    normal = normalize(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z);
                };
                Float3 emission = material.emissive.xyz();
                $if (material.textures1.x != kInvalidTexture) {
                    Float3 sampled = textures.tex2d(material.textures1.x).sample(uv).xyz();
                    emission *= pow(max(sampled, make_float3(0.0f)), 2.2f);
                };
                radiance += throughput * emission;

                $if (!wrote_guide) {
                    guide.write(pixel, make_float4(normal, length(position - camera_position)));
                    wrote_guide = true;
                };

                Float3 view_direction = -ray->direction();
                $for (light_index, kMaxLights) {
                    $if (light_index < light_count) {
                        Var<GpuLight> light = lights.read(light_index);
                        UInt light_type = cast<uint>(light.position_type.w);
                        Float3 light_direction = light.direction_range.xyz();
                        Float max_distance = def(1.0e20f);
                        Float attenuation = def(1.0f);
                        $if (light_type != 0u) {
                            Float3 to_light = light.position_type.xyz() - position;
                            max_distance = length(to_light);
                            light_direction = to_light / max(max_distance, 1.0e-4f);
                            Float range_weight = clamp(
                                    1.0f - max_distance / max(light.direction_range.w, 1.0e-3f),
                                    0.0f,
                                    1.0f);
                            attenuation = range_weight * range_weight /
                                          max(max_distance * max_distance, 0.01f);
                            $if (light_type == 2u) {
                                Float cone = dot(-light_direction, normalize(light.direction_range.xyz()));
                                attenuation *= smoothstep(light.spot_cosines.y,
                                                          light.spot_cosines.x,
                                                          cone);
                            };
                        };
                        Var<Ray> shadow_ray = make_ray(offset_ray_origin(position, normal),
                                                       light_direction,
                                                       0.0f,
                                                       max_distance - 1.0e-3f);
                        Bool occluded = accel.intersect_any(
                                shadow_ray, {.visibility_mask = kShadowVisibility});
                        $if (!occluded) {
                            Float3 light_radiance = light.color_intensity.xyz() *
                                                   light.color_intensity.w * attenuation;
                            radiance += throughput * direct_brdf(normal,
                                                                 view_direction,
                                                                 light_direction,
                                                                 light_radiance,
                                                                 albedo.xyz(),
                                                                 metallic,
                                                                 roughness,
                                                                 material.pbr.z);
                        };
                    };
                };

                Var<Onb> basis = make_onb(normal);
                Float3 diffuse_direction = basis->to_world(cosine_sample(
                        make_float2(random_float(state), random_float(state))));
                Float3 reflected = reflect(ray->direction(), normal);
                Float3 glossy_direction = normalize(lerp(
                        reflected,
                        basis->to_world(cosine_sample(make_float2(
                                random_float(state), random_float(state)))),
                        roughness * roughness));
                Bool choose_metal = random_float(state) < metallic;
                Float3 next_direction = ite(choose_metal, glossy_direction, diffuse_direction);
                Float3 f0 = lerp(make_float3(0.02f + 0.06f * material.pbr.z), albedo.xyz(), metallic);
                throughput *= ite(choose_metal, fresnel(max(dot(normal, next_direction), 0.0f), f0), albedo.xyz());
                ray = make_ray(offset_ray_origin(position, normal), next_direction);

                $if (depth >= 2u) {
                    Float survival = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05f, 0.98f);
                    $if (random_float(state) > survival) { $break; };
                    throughput /= survival;
                };
            };

            Float4 previous = accumulation.read(pixel);
            accumulation.write(pixel, previous + make_float4(max(radiance, make_float3(0.0f)), 1.0f));
            seeds.write(pixel, make_uint4(state, 0u, 0u, 0u));
        };

        Kernel2D tone_kernel = [](ImageFloat accumulation,
                                  ImageFloat guide,
                                  BufferUInt output,
                                  Float exposure,
                                  UInt denoise,
                                  UInt2 resolution) noexcept {
            UInt2 pixel = dispatch_id().xy();
            Float4 center_accum = accumulation.read(pixel);
            Float3 color = center_accum.xyz() / max(center_accum.w, 1.0f);
            $if ((denoise != 0u) & (center_accum.w < 128.0f)) {
                Float4 center_guide = guide.read(pixel);
                Float3 weighted = def(make_float3(0.0f));
                Float total_weight = def(0.0f);
                $for (offset_y, 5u) {
                    $for (offset_x, 5u) {
                        Int2 offset = make_int2(cast<int>(offset_x) - 2, cast<int>(offset_y) - 2);
                        Int2 sample_position = clamp(make_int2(pixel) + offset,
                                                     make_int2(0),
                                                     make_int2(resolution) - 1);
                        Float4 sample_accum = accumulation.read(make_uint2(sample_position));
                        Float4 sample_guide = guide.read(make_uint2(sample_position));
                        Float normal_weight = pow(max(dot(center_guide.xyz(), sample_guide.xyz()), 0.0f), 32.0f);
                        Float depth_weight = exp(-abs(center_guide.w - sample_guide.w) /
                                                 max(0.05f, center_guide.w * 0.03f));
                        Float spatial_weight = exp(-0.35f * dot(make_float2(offset), make_float2(offset)));
                        Float weight = normal_weight * depth_weight * spatial_weight;
                        weighted += sample_accum.xyz() / max(sample_accum.w, 1.0f) * weight;
                        total_weight += weight;
                    };
                };
                color = weighted / max(total_weight, 1.0e-4f);
            };
            color *= exposure;
            color = clamp((color * (2.51f * color + 0.03f)) /
                          (color * (2.43f * color + 0.59f) + 0.14f),
                          0.0f,
                          1.0f);
            color = pow(color, 1.0f / 2.2f);
            UInt red = cast<uint>(round(color.x * 255.0f));
            UInt green = cast<uint>(round(color.y * 255.0f));
            UInt blue = cast<uint>(round(color.z * 255.0f));
            UInt packed = red | (green << 8u) | (blue << 16u) | (255u << 24u);
            output.write(pixel.y * resolution.x + pixel.x, packed);
        };

        Kernel2D product_kernel = [=](BufferUInt rgb,
                                      BufferFloat linear_depth,
                                      BufferFloat4 world_normal,
                                      BufferUInt output_instance_id,
                                      BufferUInt output_semantic_id,
                                      BufferUInt geometry_coverage,
                                      AccelVar accel,
                                      BindlessVar geometry,
                                      BindlessVar textures,
                                      BufferVar<GpuMaterial> materials,
                                      BufferVar<GpuLight> lights,
                                      BufferUInt instance_ids,
                                      BufferUInt semantic_ids,
                                      BufferUInt rgb_modes,
                                      UInt light_count,
                                      UInt environment_texture,
                                      UInt output_mask,
                                      UInt gaussian_active,
                                      Float3 sky_color,
                                      Float3 ground_color,
                                      Float ambient_intensity,
                                      Float environment_intensity,
                                      Float exposure,
                                      Float4x4 inverse_view_projection,
                                      Float4x4 view_matrix,
                                      Float3 camera_position) noexcept {
            set_block_size(16u, 16u, 1u);
            UInt2 pixel = dispatch_id().xy();
            UInt2 resolution_u = dispatch_size().xy();
            Float2 resolution = make_float2(resolution_u);
            Float2 uv = (make_float2(pixel) + 0.5f) / resolution;
            Float2 ndc = make_float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
            Float4 near_h = inverse_view_projection * make_float4(ndc, -1.0f, 1.0f);
            Float4 far_h = inverse_view_projection * make_float4(ndc, 1.0f, 1.0f);
            Float3 near_point = near_h.xyz() / near_h.w;
            Float3 far_point = far_h.xyz() / far_h.w;
            Var<Ray> ray = make_ray(camera_position, normalize(far_point - near_point));
            Var<TriangleHit> hit = accel.intersect(
                    ray, {.visibility_mask = kAovVisibility});

            Float3 radiance = def(make_float3(0.0f));
            Float depth_value = def(std::numeric_limits<float>::infinity());
            Float3 normal_value = def(make_float3(0.0f));
            UInt instance_value = def(0u);
            UInt semantic_value = def(0u);
            UInt coverage_value = def(0u);

            $if (hit->miss()) {
                Float3 direction = ray->direction();
                Float sky_mix = clamp(direction.z * 0.5f + 0.5f, 0.0f, 1.0f);
                radiance = lerp(ground_color, sky_color, sky_mix);
                $if (environment_texture != kInvalidTexture) {
                    Float2 environment_uv = make_float2(
                            atan2(direction.y, direction.x) / (2.0f * constants::pi) + 0.5f,
                            acos(clamp(direction.z, -1.0f, 1.0f)) / constants::pi);
                    radiance = textures.tex2d(environment_texture).sample(environment_uv).xyz() *
                               environment_intensity;
                };
            }
            $else {
                Var<Triangle> triangle = geometry.buffer<Triangle>(hit.inst * 2u + 1u).read(hit.prim);
                Var<GpuVertex> vertex0 = geometry.buffer<GpuVertex>(hit.inst * 2u).read(triangle.i0);
                Var<GpuVertex> vertex1 = geometry.buffer<GpuVertex>(hit.inst * 2u).read(triangle.i1);
                Var<GpuVertex> vertex2 = geometry.buffer<GpuVertex>(hit.inst * 2u).read(triangle.i2);
                Float4x4 model = accel.instance_transform(hit.inst);
                Float4x4 normal_transform = transpose(inverse(model));
                Float3 local_position = triangle_interpolate(
                        hit.bary, vertex0.position, vertex1.position, vertex2.position);
                Float3 position = (model * make_float4(local_position, 1.0f)).xyz();
                Float3 normal = normalize((normal_transform * make_float4(
                        triangle_interpolate(hit.bary, vertex0.normal, vertex1.normal, vertex2.normal),
                        0.0f)).xyz());
                normal = faceforward(normal, ray->direction(), normal);
                Float4 tangent4 = triangle_interpolate(
                        hit.bary, vertex0.tangent, vertex1.tangent, vertex2.tangent);
                Float3 tangent = normalize((model * make_float4(tangent4.xyz(), 0.0f)).xyz());
                tangent = normalize(tangent - normal * dot(normal, tangent));
                Float3 bitangent = normalize(cross(normal, tangent)) * tangent4.w;
                Float2 surface_uv = triangle_interpolate(
                        hit.bary, vertex0.uv, vertex1.uv, vertex2.uv);
                Float4 vertex_color = triangle_interpolate(
                        hit.bary, vertex0.color, vertex1.color, vertex2.color);
                Var<GpuMaterial> material = materials.read(hit.inst);
                Float4 albedo = material.albedo * vertex_color;
                $if (material.textures0.x != kInvalidTexture) {
                    Float4 sampled = textures.tex2d(material.textures0.x).sample(surface_uv);
                    albedo *= make_float4(
                            pow(max(sampled.xyz(), make_float3(0.0f)), 2.2f), sampled.w);
                };
                Float metallic = material.pbr.x;
                Float roughness = clamp(material.pbr.y, 0.04f, 1.0f);
                $if (material.textures0.y != kInvalidTexture) {
                    Float4 sampled = textures.tex2d(material.textures0.y).sample(surface_uv);
                    roughness = clamp(roughness * sampled.y, 0.04f, 1.0f);
                    metallic = clamp(metallic * sampled.z, 0.0f, 1.0f);
                };
                $if (material.textures0.z != kInvalidTexture) {
                    Float3 mapped =
                            textures.tex2d(material.textures0.z).sample(surface_uv).xyz() * 2.0f - 1.0f;
                    mapped = make_float3(mapped.xy() * material.pbr.w, mapped.z);
                    normal = normalize(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z);
                };

                Float sky_mix = clamp(normal.z * 0.5f + 0.5f, 0.0f, 1.0f);
                Float3 environment = lerp(ground_color, sky_color, sky_mix);
                radiance = albedo.xyz() * environment * ambient_intensity;
                Float3 emission = material.emissive.xyz();
                $if (material.textures1.x != kInvalidTexture) {
                    Float3 sampled = textures.tex2d(material.textures1.x).sample(surface_uv).xyz();
                    emission *= pow(max(sampled, make_float3(0.0f)), 2.2f);
                };
                radiance += emission;

                Float3 view_direction = -ray->direction();
                $for (light_index, kMaxLights) {
                    $if (light_index < light_count) {
                        Var<GpuLight> light = lights.read(light_index);
                        UInt light_type = cast<uint>(light.position_type.w);
                        Float3 light_direction = light.direction_range.xyz();
                        Float max_distance = def(1.0e20f);
                        Float attenuation = def(1.0f);
                        $if (light_type != 0u) {
                            Float3 to_light = light.position_type.xyz() - position;
                            max_distance = length(to_light);
                            light_direction = to_light / max(max_distance, 1.0e-4f);
                            Float range_weight = clamp(
                                    1.0f - max_distance / max(light.direction_range.w, 1.0e-3f),
                                    0.0f,
                                    1.0f);
                            attenuation = range_weight * range_weight /
                                          max(max_distance * max_distance, 0.01f);
                            $if (light_type == 2u) {
                                Float cone = dot(-light_direction,
                                                 normalize(light.direction_range.xyz()));
                                attenuation *= smoothstep(light.spot_cosines.y,
                                                          light.spot_cosines.x,
                                                          cone);
                            };
                        };
                        Var<Ray> shadow_ray = make_ray(offset_ray_origin(position, normal),
                                                       light_direction,
                                                       0.0f,
                                                       max_distance - 1.0e-3f);
                        Bool occluded = accel.intersect_any(
                                shadow_ray, {.visibility_mask = kShadowVisibility});
                        $if (!occluded) {
                            Float3 light_radiance = light.color_intensity.xyz() *
                                                   light.color_intensity.w * attenuation;
                            radiance += direct_brdf(normal,
                                                    view_direction,
                                                    light_direction,
                                                    light_radiance,
                                                    albedo.xyz(),
                                                    metallic,
                                                    roughness,
                                                    material.pbr.z);
                        };
                    };
                };

                normal_value = normal;
                depth_value = max(-(view_matrix * make_float4(position, 1.0f)).z, 0.0f);
                instance_value = instance_ids.read(hit.inst);
                semantic_value = semantic_ids.read(hit.inst);
                coverage_value = rgb_modes.read(hit.inst);
            };

            UInt linear_index = pixel.y * resolution_u.x + pixel.x;
            $if (gaussian_active != 0u) {
                geometry_coverage.write(linear_index, coverage_value);
            };
            $if ((output_mask & kRgbOutput) != 0u) {
                Float3 color = max(radiance * exposure, make_float3(0.0f));
                color = clamp((color * (2.51f * color + 0.03f)) /
                              (color * (2.43f * color + 0.59f) + 0.14f),
                              0.0f,
                              1.0f);
                color = pow(color, 1.0f / 2.2f);
                UInt red = cast<uint>(round(color.x * 255.0f));
                UInt green = cast<uint>(round(color.y * 255.0f));
                UInt blue = cast<uint>(round(color.z * 255.0f));
                rgb.write(linear_index,
                          red | (green << 8u) | (blue << 16u) | (255u << 24u));
            };
            $if ((output_mask & kDepthOutput) != 0u) {
                linear_depth.write(linear_index, depth_value);
            };
            $if ((output_mask & kNormalOutput) != 0u) {
                world_normal.write(linear_index, make_float4(normal_value, 0.0f));
            };
            $if ((output_mask & kInstanceOutput) != 0u) {
                output_instance_id.write(linear_index, instance_value);
            };
            $if ((output_mask & kSemanticOutput) != 0u) {
                output_semantic_id.write(linear_index, semantic_value);
            };
        };

        ShaderOption option{.enable_debug_info = false};
        clear_shader_ = device_->compile(clear_kernel, option);
        seed_shader_ = device_->compile(seed_kernel, option);
        trace_shader_ = device_->compile(trace_kernel, option);
        tone_shader_ = device_->compile(tone_kernel, option);
        product_shader_ = device_->compile(product_kernel, option);
}

} // namespace gobot::luisa_renderer
