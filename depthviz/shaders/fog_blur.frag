#version 410 core
// Soft blur of the fog tone (the reference photographs' soft, blurred shapes): one separable gaussian
// direction per pass, 13 taps, sigma in source texels (filmB.w). Input: the fog (r = tone) or the
// first pass's output; output R16F.
#include "common.glsl"

uniform sampler2D u_src;
uniform vec2 u_dir;     // (1, 0) or (0, 1)

in vec2 v_uv;
out vec4 f;

void main() {
    float sigma = max(u_filmB.w, 0.01);
    vec2 tx = u_dir / vec2(textureSize(u_src, 0));
    float step_ = sigma / 2.0;          // 13 taps cover +-3 sigma
    float acc = 0.0, wsum = 0.0;
    for (int i = -6; i <= 6; ++i) {
        float x = float(i) * step_;
        float w = exp(-0.5 * x * x / (sigma * sigma));
        acc += w * texture(u_src, v_uv + tx * x).r;
        wsum += w;
    }
    f = vec4(acc / wsum, 0.0, 0.0, 1.0);
}
