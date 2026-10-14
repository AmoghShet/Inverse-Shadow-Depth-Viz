#version 410 core
// P6: ASCII glyph pass with nested grids (fine for wisps, coarse for highlights), directional
// atlas rows so filaments read as strokes, palette, grain, flicker, vignette. Pure glyphs: black
// between characters. Output is linear RGB (RGBA16F); highlights exceed 1.0 for the glyph glow.
#include "common.glsl"

uniform sampler2D u_cellF;    // fine cells   (r, angle, mag, bloom)
uniform sampler2D u_cellC;    // coarse cells
uniform sampler2D u_atlasF;   // R8, 10 columns x 5 rows of glyphs at the fine cell size
uniform sampler2D u_atlasC;   // same at the coarse cell size

in vec2 v_uv;
out vec4 f;

// Returns linear colour for one cell's glyph pixel; r_out is the cell's ramp position.
vec3 shade(sampler2D atlas, vec4 c, ivec2 cid, ivec2 inCell, ivec2 cs, uint gridSalt, out float r_out) {
    float r = c.r;
    if (r < 0.0) {   // near-black haze (grade.frag): tiny flow-oriented glyphs, just above black
        r_out = 0.0;
        float h = -r;
        int col = 1 + int(floor(h * 2.99));
        int row = c.b > u_lookB.z ? 1 + int(floor(c.g / (PI * 0.25) + 0.5)) % 4 : 0;
        float ink = texelFetch(atlas, ivec2(col * cs.x + inCell.x, row * cs.y + inCell.y), 0).r;
        return vec3(0.78, 0.88, 1.0) * u_hazeA.y * (0.6 + 0.4 * h) * ink;
    }
    r_out = r;
    // where the cell is mostly camera image: plain tone ramp, little jitter (clarity); the smoke
    // and the haze keep their oriented, jittery texture
    float camShare = (u_camA.w > 0.5 && u_camA.x <= 0.0) ? c.a : 0.0;
    float levels = u_lookF.w;
    uint ch = uint(cid.x) * 73856093u ^ uint(cid.y) * 19349663u ^ gridSalt;
    float jit = (hash1(ch, u_seedU.y) - 0.5) * 2.0 * u_lookD.z * (1.0 - 0.85 * camShare);
    float li = clamp(floor(r * (levels - 1.0) + jit), 0.0, levels - 1.0);
    int col = int(floor(li * 9.0 / max(levels - 1.0, 1.0) + 0.5));
    if (col <= 0) return vec3(0.0);
    int row = 0;
    if (c.b > u_lookB.z && r < u_lookB.w && camShare < 0.5) {
        int bin = int(floor(c.g / (PI * 0.25) + 0.5)) % 4;
        row = 1 + bin;   // 1: -, 2: /, 3: |, 4: \  (velocity is y-up, as on screen)
    }
    float ink = texelFetch(atlas, ivec2(col * cs.x + inCell.x, row * cs.y + inCell.y), 0).r;
    return palette(r) * ink * (1.0 + u_look.w * r * r);
}

// Camera background: dim glyphs in the smoke's own palette, oriented by the same flow, only where
// the smoke is fainter, so the smoke always reads on top and the two look like one material.
vec3 shade_bg(vec4 c, ivec2 cid, ivec2 inCell, ivec2 cs, out float r_out) {
    float rb = c.a;
    r_out = 0.0;
    if (rb <= 0.0 || c.r >= rb) return vec3(-1.0);
    float levels = u_lookF.w;
    uint ch = uint(cid.x) * 83492791u ^ uint(cid.y) * 2654435761u ^ 0x5EA5u;
    float jit = (hash1(ch, u_seedU.y) - 0.5) * 0.8;
    float li = clamp(floor(rb * (levels - 1.0) + jit), 0.0, levels - 1.0);
    int col = int(floor(li * 9.0 / max(levels - 1.0, 1.0) + 0.5));
    if (col <= 0) return vec3(0.0);
    int row = 0;
    if (c.b > u_lookB.z) row = 1 + int(floor(c.g / (PI * 0.25) + 0.5)) % 4;
    float ink = texelFetch(u_atlasF, ivec2(col * cs.x + inCell.x, row * cs.y + inCell.y), 0).r;
    vec3 pc = palette(0.30 + rb);
    vec3 col3 = mix(vec3(dot(pc, vec3(0.3333))), pc, u_camC.z) * u_camB.w;
    r_out = rb;
    return col3 * ink;
}

void main() {
    vec2 fc = vec2(gl_FragCoord.x, u_res.y - gl_FragCoord.y);   // top-origin pixels
    ivec2 csF = ivec2(u_gridF.xy);
    ivec2 csC = ivec2(u_gridC.xy);
    ivec2 cidF = ivec2(floor(fc / vec2(csF)));
    ivec2 cidC = ivec2(floor(fc / vec2(csC)));
    ivec2 inF = ivec2(fc) - cidF * csF;
    ivec2 inC = ivec2(fc) - cidC * csC;
    vec4 F = texelFetch(u_cellF, cidF, 0);
    vec4 C = texelFetch(u_cellC, cidC, 0);

    float T = u_lookB.x, B = u_lookB.y;
    if (u_camA.w > 0.5 && u_camA.x <= 0.0) T += u_camC.w * C.a;   // the camera keeps fine glyphs (clarity)
    float w = smoothstep(T - B, T + B, C.r);

    float rF, rC;
    vec3 colF = w < 0.999 ? shade(u_atlasF, F, cidF, inF, csF, 0x1234u, rF) : vec3(0.0);
    if (w < 0.5 && u_camA.w > 0.5 && u_camA.x > 0.0) {
        float rB;
        vec3 colB = shade_bg(F, cidF, inF, csF, rB);
        if (colB.r >= 0.0) { colF = colB; rF = rB; }
    }
    vec3 colC = w > 0.001 ? shade(u_atlasC, C, cidC, inC, csC, 0x9876u, rC) : vec3(0.0);
    if (w >= 0.999) rF = 0.0;
    if (w <= 0.001) rC = 0.0;
    vec3 col = mix(colF, colC, w);
    float r = mix(rF, rC, w);

    // grain weighted to midtones, global flicker, vignette
    uint frame = uint(u_time.z);
    float n = hash1(uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * 8191u, frame);
    r = max(r, 0.0);
    float mid = 4.0 * r * (1.0 - r);
    col *= 1.0 + u_lookC.x * (n - 0.5) * 2.0 * mid;
    col *= 1.0 + u_lookC.y * (hash1(frame, 0xF11Cu) - 0.5) * 2.0;
    vec2 p = uv_to_p(v_uv);
    float dist = length(p) / length(vec2(aspect(), 1.0) * 0.5);
    col *= clamp(1.0 - u_lookC.z * dist * dist, 0.0, 1.0);

    if (u_lookE.w > 0.5) {   // grid overlay (G)
        if (inF.x == 0 || inF.y == 0) col += vec3(0.0, 0.03, 0.06);
        if (inC.x == 0 || inC.y == 0) col += vec3(0.0, 0.07, 0.12);
        if (w > 0.5) col += vec3(0.05, 0.025, 0.0);
    }
    f = vec4(col, 1.0);
}
