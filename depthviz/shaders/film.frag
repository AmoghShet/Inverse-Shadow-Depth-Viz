#version 410 core
// D4 film: the depth fog itself, as in the reference photographs. Monochrome tone from D2, softly
// blurred (fog_blur, sigma filmB.w), halation from D2b glowing over the edges, heavy film grain
// (fine + clumped, reseeded every picture tick, strong in the darks as well as the lights), dust,
// flicker, vignette. The fog tone is a display value; it is linearised here because D5 gamma-encodes.
// Output: linear RGB (RGBA16F).
#include "common.glsl"

uniform sampler2D u_fogSoft; // R16F blurred fog tone, half res
uniform sampler2D u_halo;    // R16F, 1/4 res

in vec2 v_uv;
out vec4 f;

float lattice(vec2 p, uint salt) {   // smooth value noise on a hashed lattice (grain clumps)
    vec2 i = floor(p);
    vec2 fr = p - i;
    vec2 u = fr * fr * (3.0 - 2.0 * fr);
    uint x0 = uint(int(i.x) + 65536), y0 = uint(int(i.y) + 65536);
    float a = hash1(x0 * 73856093u ^ y0 * 19349663u, salt);
    float b = hash1((x0 + 1u) * 73856093u ^ y0 * 19349663u, salt);
    float c = hash1(x0 * 73856093u ^ (y0 + 1u) * 19349663u, salt);
    float d = hash1((x0 + 1u) * 73856093u ^ (y0 + 1u) * 19349663u, salt);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

void main() {
    float v = texture(u_fogSoft, v_uv).r;
    v += u_haloA.y * texture(u_halo, v_uv).r;          // halation: bright fog glows over the edges
    v = clamp(v, 0.0, 1.2);
    vec3 col = palette(min(v, 1.0)) * (1.0 + 0.5 * max(v - 1.0, 0.0));
    // print range: blacks never quite black, whites never quite white (as in the references), so the
    // grain reads in both
    col = u_filmC.x + (u_filmC.y - u_filmC.x) * col;

    // film grain: fine (per pixel) + clumped (lattice at grain_size), reseeded every tick, in display
    // space; the darks keep most of it (filmA.w) as in the references
    uint tick = u_seedU.z;
    float s = max(1.0, u_res.y / 1080.0);
    vec2 px = gl_FragCoord.xy / s;
    float fine = hash1(uint(px.x) + uint(px.y) * 8191u, tick, 0x6A1Du) - 0.5;
    float clump = lattice(px / max(u_filmA.y, 0.5), tick ^ 0xC1u) - 0.5;
    float g = mix(fine, clump * 1.6, u_filmA.z);
    float amt = u_filmA.x * mix(u_filmA.w, 1.0, smoothstep(0.05, 0.8, v));
    col += vec3(g * amt * 2.0);
    // dust specks (2x2 at 1080p), bright or dark
    uvec2 blk = uvec2(gl_FragCoord.xy / max(1.0, floor(u_res.y / 540.0 + 0.5)));
    if (hash1(blk.x * 2654435761u ^ blk.y * 40503u, tick, 0xD057u) < u_filmB.z) {
        col = hash1(blk.x ^ 0x51u, blk.y, tick) < 0.5 ? vec3(0.0) : palette(1.0);
    }
    // flicker, vignette
    col *= 1.0 + u_filmB.x * (hash1(tick, 0xF11Cu) - 0.5) * 2.0;
    vec2 p = uv_to_p(v_uv);
    float dist = length(p) / length(vec2(aspect(), 1.0) * 0.5);
    col *= clamp(1.0 - u_filmB.y * dist * dist, 0.0, 1.0);
    col = pow(clamp(col, 0.0, 1.0), vec3(2.2));        // display tone -> linear (D5 encodes)
    f = vec4(col, 1.0);
}
