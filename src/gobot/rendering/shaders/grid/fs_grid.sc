$input uv, out_camPos

#include "../common.sh"


void main()
{
    gl_FragColor = gridColor(uv, out_camPos);
};
