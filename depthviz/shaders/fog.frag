#version 410 core
// D2 fog (half screen res). Cover-crop into camera space, 2.5D parallax (one depth tap decides the
// shift), transmittance T = exp(-k d) with a slow drifting density swirl, a seeded window/door-like
// light behind the figure, tone v = mix(near, far * L, 1 - T), exposure with a soft shoulder, gamma.
// The far tone is modulated by the same swirl, so the haze drifts. Polarity is only the (near, far) tone pair, so a flip is a crossfade of the pair plus an exposure
// push through white or black (director). Output RGBA16F: r = tone v (0..~1), gb = depth gradient
// (screen orientation), a = mask.
#include "common.glsl"

uniform sampler2D u_filt;   // RGBA16F (depth, gx, gy, mask), camera space

in vec2 v_uv;
out vec4 f;

void main() {
    vec2 cuv = cam_uv(v_uv);
    vec4 s;
    if (u_camA.z > 0.5) {
        float d0 = texture(u_filt, cuv).r;
        vec2 uvc = cuv + (u_fogD.z - d0) * u_fogD.xy * u_camA.xy;
        s = texture(u_filt, clamp(uvc, vec2(0.001), vec2(0.999)));
    } else {
        s = vec4(1.0, 0.0, 0.0, 0.0);   // no depth yet: empty room
    }
    // Depth -> tone is continuous, like the depth map itself (the references: the near shoulder darker than
    // the face, walls fading into the haze), not a near/far switch: F(d) = (1 - exp(-k d^g)) / (1 - exp(-k)),
    // 0 at the nearest point, 1 at the farthest; k = how fast the fog thickens with distance, g = curve
    // (bass bends it). The inverted look (V key) uses 0.4x the density so near shapes stay pale and only the
    // far room sinks into the black (keyness crossfades it).
    float g = max(u_fogF.w, 0.2) * u_fogE.y;
    vec2 p = uv_to_p(v_uv);
    // density swirl: slow two-octave noise drifting through the fog (breaks up flat haze)
    float sw = 0.0;
    if (u_fogE.w > 0.0) {
        vec3 q = vec3(p * u_fogF.y, u_time.x * u_fogF.z);
        sw = 0.65 * vnoise3(q) + 0.35 * vnoise3(q * 2.3 + 7.1);
    }
    float k = max(u_fogA.x * mix(0.4, 1.0, u_look.x) * max(0.05, 1.0 + u_fogE.w * sw), 0.05);
    float F = (1.0 - exp(-k * pow(clamp(s.r, 0.0, 1.0), g))) / (1.0 - exp(-k));
    // light field: a soft-edged bright side (window / door) plus a glow blob, behind the figure
    vec2 c = u_fogC.xy;
    vec2 dir = u_fogC.zw;
    float w = max(u_fogD.w, 1e-3);
    float side = smoothstep(-w, w, dot(p - c, dir));
    float rr = max(u_fogE.x, 1e-3);
    float blob = exp(-dot(p - c, p - c) / (2.0 * rr * rr));
    float L = (u_fogB.w + u_fogB.y * side + (u_fogB.z + u_fogE.z) * blob) * max(0.0, 1.0 + 1.5 * u_fogE.w * sw);
    // fog close to the camera is lit by the room's ambient only, the window light reaches the far fog
    float Leff = mix(0.5 * u_fogB.w, L, smoothstep(0.0, 0.7, s.r));
    float v = mix(u_fogA.y, u_fogA.z * Leff, F);
    // exposure: multiplicative (blackout -> 0, whiteout -> large), then a soft shoulder above the
    // knee so highlights roll off toward 1 without lifting the shadows
    v = max(v, 0.0) * u_fogB.x;
    float kn = clamp(u_fogF.x, 0.3, 0.95);
    if (v > kn) v = kn + (1.0 - kn) * (1.0 - exp(-(v - kn) / (1.0 - kn)));
    v = pow(max(v, 0.0), u_fogA.w);
    // gradient: camera texels are square, so the cover crop keeps directions; scale is per 1% height
    f = vec4(v, s.g, s.b, s.a);
}
