#version 410 core
// P4 step 1: 4x4 box downsample (4 bilinear taps) with a soft threshold.
// MODE_DENSITY: input is raw density, thresholded after the tone curve.
// MODE_COLOR:   input is the linear RGB frame, thresholded on max channel (glyph glow).
#include "common.glsl"

uniform sampler2D u_src;
uniform float u_threshold;

in vec2 v_uv;
out vec4 f;

void main() {
    vec2 tx = 1.0 / vec2(textureSize(u_src, 0));
    vec4 s = texture(u_src, v_uv + tx * vec2(-1.0, -1.0))
           + texture(u_src, v_uv + tx * vec2( 1.0, -1.0))
           + texture(u_src, v_uv + tx * vec2(-1.0,  1.0))
           + texture(u_src, v_uv + tx * vec2( 1.0,  1.0));
    s *= 0.25;
#ifdef MODE_DENSITY
    float v = tone(s.r);
    float b = max(v - u_threshold, 0.0) / max(1.0 - u_threshold, 1e-3);
    f = vec4(b * b + 0.5 * b);
#else
    float m = max(max(s.r, s.g), s.b);
    float k = max(m - u_threshold, 0.0) / max(m, 1e-4);
    f = vec4(s.rgb * k, 1.0);
#endif
}
