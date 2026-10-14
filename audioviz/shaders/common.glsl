// Shared declarations: Frame UBO, integer hashes, gradient noise, curl, tone curve, palette.
// Included by every program; no #version here.

#include "frame_ubo"

#define PI 3.14159265359

// ---- integer hashes (identical on every vendor, unlike fract(sin(x))) -----------------------
uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
uint hash2u(uint a, uint b) { return pcg(a ^ pcg(b)); }
uint hash3u(uint a, uint b, uint c) { return pcg(a ^ pcg(b ^ pcg(c))); }
float u2f(uint h) { return float(h >> 8u) * (1.0 / 16777216.0); }   // [0,1)
float hash1(uint a, uint b) { return u2f(hash2u(a, b)); }
float hash1(uint a, uint b, uint c) { return u2f(hash3u(a, b, c)); }

// ---- 3D gradient noise (Perlin style) with an integer lattice hash --------------------------
vec3 lattice_grad(ivec3 c) {
    uint h = hash3u(uint(c.x) * 73856093u, uint(c.y) * 19349663u, uint(c.z) * 83492791u ^ u_seedU.x);
    // 12 edge directions of a cube, picked by the hash
    uint k = h % 12u;
    vec3 g = k < 4u ? vec3((k & 1u) == 0u ? 1.0 : -1.0, (k & 2u) == 0u ? 1.0 : -1.0, 0.0)
           : k < 8u ? vec3((k & 1u) == 0u ? 1.0 : -1.0, 0.0, (k & 2u) == 0u ? 1.0 : -1.0)
                    : vec3(0.0, (k & 1u) == 0u ? 1.0 : -1.0, (k & 2u) == 0u ? 1.0 : -1.0);
    return g;
}

float noise3(vec3 x) {
    vec3 fl = floor(x);
    ivec3 i = ivec3(fl);
    vec3 f = x - fl;
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float a = dot(lattice_grad(i + ivec3(0, 0, 0)), f - vec3(0, 0, 0));
    float b = dot(lattice_grad(i + ivec3(1, 0, 0)), f - vec3(1, 0, 0));
    float c = dot(lattice_grad(i + ivec3(0, 1, 0)), f - vec3(0, 1, 0));
    float d = dot(lattice_grad(i + ivec3(1, 1, 0)), f - vec3(1, 1, 0));
    float e = dot(lattice_grad(i + ivec3(0, 0, 1)), f - vec3(0, 0, 1));
    float g = dot(lattice_grad(i + ivec3(1, 0, 1)), f - vec3(1, 0, 1));
    float h = dot(lattice_grad(i + ivec3(0, 1, 1)), f - vec3(0, 1, 1));
    float k = dot(lattice_grad(i + ivec3(1, 1, 1)), f - vec3(1, 1, 1));
    return mix(mix(mix(a, b, u.x), mix(c, d, u.x), u.y), mix(mix(e, g, u.x), mix(h, k, u.x), u.y), u.z);
}

// Curl of a scalar potential: divergence-free 2D velocity. Divided by nothing: caller scales.
vec2 curl(vec2 p, float z) {
    const float e = 0.01;
    float dx = noise3(vec3(p.x + e, p.y, z)) - noise3(vec3(p.x - e, p.y, z));
    float dy = noise3(vec3(p.x, p.y + e, z)) - noise3(vec3(p.x, p.y - e, z));
    return vec2(dy, -dx) / (2.0 * e);
}

// ---- coordinate spaces ----------------------------------------------------------------------
// uv: [0,1]^2 over the frame. p: aspect-corrected, centred, height 1 (x in [-aspect/2, aspect/2]).
float aspect() { return u_res.z; }
vec2 uv_to_p(vec2 uv) { return (uv - 0.5) * vec2(aspect(), 1.0); }
vec2 p_to_uv(vec2 p) { return p / vec2(aspect(), 1.0) + 0.5; }
vec2 pvel_to_uv(vec2 v) { return v / vec2(aspect(), 1.0); }

// ---- tone curve -----------------------------------------------------------------------------
// tone(d) = pow(smoothstep(knee, 1, 1 - exp(-d * exposure)), gamma)
float tone(float d) {
    float x = 1.0 - exp(-max(d, 0.0) * u_look.x);
    return pow(smoothstep(u_look.y, 1.0, x), u_look.z);
}

// ---- camera level (shared by the background glyphs and the smoke the feed emits) -------------
// Auto-levels (black/white points from the camera process), a contrast crush that drops the
// shadows to black so only lit shapes remain, a mip blur so it reads as shape rather than photo.
float cam_level(sampler2D cam, vec2 uv) {
    vec2 cuv = (uv - 0.5) * u_camB.xy + 0.5;
    float g = textureLod(cam, cuv, u_camC.x).r;
    g = clamp((g - u_camA.y) / max(u_camA.z - u_camA.y, 0.02), 0.0, 1.0);
    float c = u_camC.y;
    g = smoothstep(0.5 * c, 1.0 - 0.15 * c, g);
    return pow(g, u_camB.z);
}

// ---- palette: 4-stop gradient ---------------------------------------------------------------
vec3 palette(float r) {
    vec3 c = u_palette[0].rgb;
    for (int i = 1; i < 4; ++i) {
        float a = u_palette[i - 1].a, b = u_palette[i].a;
        c = mix(c, u_palette[i].rgb, clamp((r - a) / max(b - a, 1e-4), 0.0, 1.0));
    }
    return c;
}
