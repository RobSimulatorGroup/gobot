#version 460 core

#include "gl_buffer_declation.h"
#include "grid_parameters.h"

layout (location=0) out vec2 uv;
layout (location=1) out vec2 out_camPos;


void main()
{
    mat4 MVP = proj * view;

    int idx = indices[gl_VertexID];
    vec3 position = pos[idx] * gridSize;

    position.x += cameraPos.x;
    position.z += cameraPos.z;

    out_camPos = cameraPos.xz;

    gl_Position = MVP * vec4(position, 1.0);
    uv = position.xz;
}
