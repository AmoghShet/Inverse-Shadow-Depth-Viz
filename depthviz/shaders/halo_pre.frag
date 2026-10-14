#version 410 core
// D2b halation step 1: 4-tap box downsample of the fog tone with a soft threshold (highlights only).
#include "common.glsl"
uniform sampler2D u_src;
in vec2 v_uv;
out vec4 f;
void main() {
    vec2 tx = 1.0 / vec2(textureSize(u_src, 0));
    float s = 0.25 * (texture(u_src, v_uv + tx * vec2(-1.0, -1.0)).r + texture(u_src, v_uv + tx * vec2(1.0, -1.0)).r
                    + texture(u_src, v_uv + tx * vec2(-1.0, 1.0)).r + texture(u_src, v_uv + tx * vec2(1.0, 1.0)).r);
    float t = u_haloA.x;
    float b = max(s - t, 0.0) / max(1.0 - t, 1e-3);
    f = vec4(b, 0.0, 0.0, 1.0);
}
