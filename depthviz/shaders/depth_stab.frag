#version 410 core
// D1 stabilise (per picture tick, 640x480, camera space). EMA toward the newest depth with
// a = 1 - exp(-dt_tick / tau); where the depth jumps by more than the edge-reset threshold (an edge
// moved) a = 0.85, so motion does not smear.
// The figure edge comes from the mask, not the depth (monocular models blur depth across a
// silhouette). Outside the mask the depth is at least figDepth + margin (plan) and, near the
// figure, the far value of a 16-tap ring (grey dilation of the background: the halo is replaced by
// the room behind it); inside the mask, near the edge, it is the near value of the ring's inside taps.
// The sharp mask edge becomes the silhouette.
// Output RG16F: r = depth (0 near .. 1 far), g = mask (EMA'd the same way).
#include "common.glsl"

uniform sampler2D u_depth;   // R16F 320x240, GL rows, mirrored camera space
uniform sampler2D u_mask;    // R8
uniform sampler2D u_prev;    // RG16F previous output
uniform float u_reset;       // 1 = take the new depth as is

in vec2 v_uv;
out vec4 f;

void main() {
    float dn = texture(u_depth, v_uv).r;
    float m = texture(u_mask, v_uv).r;
    if (u_depthB.z > 0.5) {
        vec2 tx = 1.0 / vec2(textureSize(u_depth, 0));
        float dbg = dn, dfg = dn;
        float nin = 0.0, nout = 0.0;
        for (int r = 1; r <= 2; ++r) {
            float rad = 6.0 * float(r);
            for (int k = 0; k < 8; ++k) {
                float a = float(k) * (PI * 0.25) + 0.39 * float(r);
                vec2 o = vec2(cos(a), sin(a)) * rad * tx;
                float dt = texture(u_depth, v_uv + o).r;
                float mt = texture(u_mask, v_uv + o).r;
                if (mt < 0.5) { dbg = max(dbg, dt); nout += 1.0; }
                else { dfg = min(dfg, dt); nin += 1.0; }
            }
        }
        float fig = u_depthA.z;
        float dout = max(dn, fig + u_depthA.w);
        if (nin > 0.0) dout = max(dout, dbg);      // near the figure: the room behind, not the halo
        // inside: erode only near the silhouette (where the halo is); deeper inside keep the model's own
        // depth so the figure keeps its shape (face / shoulders), not a flat cut-out
        float din = (nin > 0.0 && nout > 0.0) ? min(dn, dfg) : dn;
        dn = mix(dout, din, smoothstep(0.35, 0.65, m));
    } else {
        m = 0.0;
    }
    vec2 prev = texture(u_prev, v_uv).rg;
    float a = u_depthA.x;
    if (abs(dn - prev.r) > u_depthA.y) a = max(a, 0.85);
    if (u_reset > 0.5) a = 1.0;
    f = vec4(mix(prev.r, dn, a), mix(prev.g, m, max(a, 0.5)), 0.0, 1.0);
}
