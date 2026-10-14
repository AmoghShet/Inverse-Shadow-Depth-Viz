#version 410 core
// D5 present. View 0 = the film frame; 1 = stabilised depth; 2 = fog tone (+ halation);
// 3 = mask over dim depth. Soft shoulder above 0.8, then gamma 2.2 encode.
#include "common.glsl"

uniform sampler2D u_frame;   // RGBA16F linear film frame
uniform sampler2D u_filt;    // RGBA16F (depth, gx, gy, mask), camera space
uniform sampler2D u_fog;     // RGBA16F (v, ...), half res
uniform sampler2D u_halo;    // R16F
uniform int u_view;

in vec2 v_uv;
out vec4 f;

vec3 shoulder(vec3 c) {
    c = max(c, 0.0);
    vec3 hi = 0.8 + 0.2 * (1.0 - exp(-(c - 0.8) / 0.2));
    return mix(c, hi, step(0.8, c));
}

void main() {
    vec3 col;
    vec2 cuv = cam_uv(v_uv);
    if (u_view == 1) {
        col = vec3(texture(u_filt, cuv).r);
    } else if (u_view == 2) {
        col = vec3(texture(u_fog, v_uv).r + u_haloA.y * texture(u_halo, v_uv).r);
    } else if (u_view == 3) {
        vec4 s = texture(u_filt, cuv);
        col = vec3(0.25 * s.r) + vec3(0.9, 0.55, 0.2) * s.a;
    } else {
        col = texture(u_frame, v_uv).rgb;
    }
    col = shoulder(col);
    f = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);
}
