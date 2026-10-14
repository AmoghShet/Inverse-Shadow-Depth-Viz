// Shared declarations: Frame UBO, integer hashes, gradient noise (vnoise3), coordinate spaces,
// cover crop into camera space, palette. Included by every program; no #version here.
// NOTE: never name a function noise3: it clashes with the GLSL 4.1 built-in on Apple drivers.

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

// ---- 3D gradient noise with an integer lattice hash -----------------------------------------
vec3 lattice_grad(ivec3 c) {
    uint h = hash3u(uint(c.x) * 73856093u, uint(c.y) * 19349663u, uint(c.z) * 83492791u ^ u_seedU.x);
    uint k = h % 12u;
    return k < 4u ? vec3((k & 1u) == 0u ? 1.0 : -1.0, (k & 2u) == 0u ? 1.0 : -1.0, 0.0)
         : k < 8u ? vec3((k & 1u) == 0u ? 1.0 : -1.0, 0.0, (k & 2u) == 0u ? 1.0 : -1.0)
                  : vec3(0.0, (k & 1u) == 0u ? 1.0 : -1.0, (k & 2u) == 0u ? 1.0 : -1.0);
}

float vnoise3(vec3 x) {
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

// ---- coordinate spaces ----------------------------------------------------------------------
// uv: [0,1]^2 over the frame. p: aspect-corrected, centred, height 1 (x in [-aspect/2, aspect/2]).
float aspect() { return u_res.z; }
vec2 uv_to_p(vec2 uv) { return (uv - 0.5) * vec2(aspect(), 1.0); }
vec2 p_to_uv(vec2 p) { return p / vec2(aspect(), 1.0) + 0.5; }
// screen uv -> camera uv (cover crop: the 4:3 camera fills the frame, no stretch)
vec2 cam_uv(vec2 uv) { return (uv - 0.5) * u_camA.xy + 0.5; }

// ---- palette: 4-stop gradient ---------------------------------------------------------------
vec3 palette(float r) {
    vec3 c = u_palette[0].rgb;
    for (int i = 1; i < 4; ++i) {
        float a = u_palette[i - 1].a, b = u_palette[i].a;
        c = mix(c, u_palette[i].rgb, clamp((r - a) / max(b - a, 1e-4), 0.0, 1.0));
    }
    return c;
}
