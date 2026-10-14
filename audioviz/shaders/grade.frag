#version 410 core
// P5: one fragment per ASCII cell. Supersample the density over the cell, tone it, add bloom,
// and record the flow angle and magnitude.
// Output RGBA16F: r = ramp position (0..1), g = flow angle folded to [0, pi), b = flow magnitude,
// a = camera: glyph-layer level (layer mode, camA.x > 0) or the camera's share of this cell's
//     density (camera-as-smoke mode), used to keep the feed in fine glyphs.
// Cell textures use top-origin rows: texel (x, y) is column x, row y counted from the top.
#include "common.glsl"

uniform sampler2D u_dens;
uniform sampler2D u_bloom;
uniform sampler2D u_vel;
uniform sampler2D u_cam;  // R8 gray camera frame, mirrored, GL rows (1x1 black when off)
uniform sampler2D u_ghost; // RG camera ghost buffer (live, hold)
uniform vec4 u_grid;      // cell w, h (px), cols, rows

out vec4 f;

void main() {
    vec2 cell = floor(gl_FragCoord.xy);
    vec2 fb = u_res.xy;
    vec2 cs = u_grid.xy;
    // pixel rect of this cell, top-origin; convert to GL uv (bottom-origin)
    vec2 px0 = cell * cs;
    int nx = clamp(int(cs.x * 0.5 + 0.5), 1, 8);
    int ny = clamp(int(cs.y * 0.5 + 0.5), 1, 16);
    float sum = 0.0, mx = 0.0;
    for (int j = 0; j < 16; ++j) {
        if (j >= ny) break;
        for (int i = 0; i < 8; ++i) {
            if (i >= nx) break;
            vec2 px = px0 + (vec2(i, j) + 0.5) * cs / vec2(nx, ny);
            vec2 uv = vec2(px.x / fb.x, 1.0 - px.y / fb.y);
            float d = texture(u_dens, uv).r;
            sum += d;
            mx = max(mx, d);
        }
    }
    float mean = sum / float(nx * ny);
    float d = mix(mean, mx, u_lookD.w);
    vec2 cuv = vec2((px0.x + 0.5 * cs.x) / fb.x, 1.0 - (px0.y + 0.5 * cs.y) / fb.y);
    float bl = texture(u_bloom, cuv).r * u_lookD.x;
    // bloom lifts existing smoke (multiplicative); it never draws glyphs into empty space, which
    // turned bright clusters into large solid blobs
    float r0 = tone(d);
    float r = clamp(r0 * (1.0 + 0.6 * bl), 0.0, 1.0);
    vec2 v = texture(u_vel, cuv).xy;
    float ang = atan(v.y, v.x);
    if (ang < 0.0) ang += PI;
    if (ang >= PI) ang -= PI;
    float bg = 0.0;
    if (u_camA.w > 0.5) {
        if (u_camA.x > 0.0) {
            bg = cam_level(u_cam, cuv) * u_camA.x;
        } else {
            vec2 gh = texture(u_ghost, cuv).rg;
            float camd = u_camD.x * max(gh.r, u_camE.z * gh.g);
            bg = clamp(camd / max(d, 1e-4), 0.0, 1.0);
        }
    }
    // Near-black haze in empty cells, encoded as a negative ramp value (-h): a slow two-octave
    // noise field drifting with the drift vector, thinned per cell, so the black reads as faint
    // smoke. Smoke always wins: only cells the smoke and the camera leave empty get haze.
    if (u_hazeA.x > 0.0 && r < 0.035) {
        vec2 hp = uv_to_p(cuv) * u_hazeA.z - u_simE.xy * u_time.x * 2.0;
        float n = 0.65 * noise3(vec3(hp, u_time.x * u_hazeA.w))
                + 0.35 * noise3(vec3(hp * 2.7 + 11.0, u_time.x * u_hazeA.w * 2.3));
        float h = smoothstep(-0.1, 0.45, n);
        uint ch = uint(cell.x) * 2654435761u ^ uint(cell.y) * 40503u ^ 0xA2E1u;
        if (hash1(ch, u_seedU.y) < u_hazeA.x * h) r = -max(h, 0.05);
    }
    f = vec4(r, ang, length(v), bg);
}
