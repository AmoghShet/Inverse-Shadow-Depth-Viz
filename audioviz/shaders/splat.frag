#version 410 core
// P2: gaussian point, blended ONE,ONE into the density buffer (never cleared; P3 decays it).
#include "common.glsl"

in float v_w;
out vec4 f;

void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float g = exp(-dot(d, d) * u_simG.w);
    f = vec4(v_w * g * u_simE.z);
}
