$output uv, out_camPos

#include "../common.sh"

// extents of grid in world coordinates
float gridSize = 100.0;

// size of one cell
float gridCellSize = 0.025;

// color of thin lines
vec4 gridColorThin = vec4(0.5, 0.5, 0.5, 1.0);

// color of thick lines (every tenth line)
vec4 gridColorThick = vec4(0.0, 0.0, 0.0, 1.0);

// minimum number of pixels between cell lines before LOD switch should occur.
const float gridMinPixelsBetweenCells = 2.0;


void main()
{
//	int idx = indices[gl_VertexID];
	vec3 position = vec3(0.0, 0.0, 0.0);

//	position.x += cameraPos.x;
//	position.z += cameraPos.z;

	out_camPos = position.xz;

	gl_Position = u_modelViewProj * vec4(position, 1.0);
	uv = position.xz;
}
