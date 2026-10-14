#version 410 core
// D1b filter (640x480): 5x5 cross-bilateral on the stabilised depth (spatial sigma in texels,
// range sigma in depth units, the mask edge as a guide: taps across the silhouette are ignored),
// optional sharpening toward the unfiltered value (onsets), Sobel gradient, feathered mask.
// Output RGBA16F: r = depth, g = d(depth)/dx, b = d(depth)/dy (per 1% of the frame height), a = mask.
#include "common.glsl"

uniform sampler2D u_stab;   // RG16F (depth, mask)

in vec2 v_uv;
out vec4 f;

void main() {
    vec2 tx = 1.0 / vec2(textureSize(u_stab, 0));
    vec2 c = texture(u_stab, v_uv).rg;
    float ss = max(u_depthB.x, 0.5), sr = max(u_depthB.y, 1e-3);
    float wsum = 0.0, dsum = 0.0, msum = 0.0, mw = 0.0;
    for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
            vec2 s = texture(u_stab, v_uv + vec2(i, j) * tx).rg;
            float r2 = float(i * i + j * j);
            float w = exp(-r2 / (2.0 * ss * ss)) * exp(-(s.r - c.r) * (s.r - c.r) / (2.0 * sr * sr));
            w *= 1.0 - clamp(abs(s.g - c.g) * 1.5, 0.0, 0.95);
            wsum += w;
            dsum += w * s.r;
            float wm = exp(-r2 / (2.0 * 1.5 * 1.5));
            msum += wm * s.g;
            mw += wm;
        }
    }
    float d = dsum / max(wsum, 1e-5);
    d = clamp(d + u_depthB.w * (c.r - d), 0.0, 1.0);
    // Sobel on the stabilised depth
    float tl = texture(u_stab, v_uv + vec2(-1, 1) * tx).r, t = texture(u_stab, v_uv + vec2(0, 1) * tx).r;
    float tr = texture(u_stab, v_uv + vec2(1, 1) * tx).r, l = texture(u_stab, v_uv + vec2(-1, 0) * tx).r;
    float r = texture(u_stab, v_uv + vec2(1, 0) * tx).r, bl = texture(u_stab, v_uv + vec2(-1, -1) * tx).r;
    float b = texture(u_stab, v_uv + vec2(0, -1) * tx).r, br = texture(u_stab, v_uv + vec2(1, -1) * tx).r;
    float gx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
    float gy = (tl + 2.0 * t + tr) - (bl + 2.0 * b + br);
    vec2 g = vec2(gx, gy) / 8.0 * (0.01 / tx.y);   // depth per 1% of the frame height
    f = vec4(d, g, msum / max(mw, 1e-5));
}
