#version 410 core
// P1: particle update. State texel = (u, v, age 0..1 alive / >=1 dead, per-particle hash).
// Emission is a ring over particle indices: [emitStart, emitStart+emitCount) respawn at weighted
// emitters, the next burstCount respawn at the burst emitter (kick puffs). Everything else advects.
#include "common.glsl"

uniform sampler2D u_state;
uniform sampler2D u_vel;

out vec4 f;

vec2 gauss2(uint a, uint b) {
    float u1 = max(hash1(a, b, 0x9E37u), 1e-6);
    float u2 = hash1(a, b, 0x7F4Au);
    float r = sqrt(-2.0 * log(u1));
    return r * vec2(cos(2.0 * PI * u2), sin(2.0 * PI * u2));
}

void main() {
    ivec2 ij = ivec2(gl_FragCoord.xy);
    int S = textureSize(u_state, 0).x;
    int idx = ij.y * S + ij.x;
    int N = int(u_seedU.w);
    vec4 s = texelFetch(u_state, ij, 0);
    float dt = u_time.y;
    uint step_ = u_seedU.z;
    uint uidx = uint(idx);

    int rel = idx - u_emit.x;
    if (rel < 0) rel += N;
    bool normal = rel < u_emit.y;
    bool burst = !normal && rel < u_emit.y + u_emit.z;

    if (normal || burst) {
        int n = int(u_emitB.x);
        int e = u_emit.w;
        if (normal) {
            float total = 0.0;
            for (int i = 0; i < 8; ++i) { if (i < n) total += u_emitters[i].z; }
            float r = hash1(uidx, step_, 0x51EDu) * total;
            e = 0;
            float acc = 0.0;
            for (int i = 0; i < 8; ++i) {
                if (i >= n) break;
                acc += u_emitters[i].z;
                e = i;
                if (r < acc) break;
            }
        }
        float rad = u_emitters[e].w * u_simG.y * (burst ? u_emitB.w : 1.0);
        vec2 p = u_emitters[e].xy + rad * gauss2(uidx, step_);
        // the per-particle hash is re-drawn on respawn so fur strands do not repeat
        float h = hash1(uidx, step_, u_seedU.x);
        f = vec4(p_to_uv(p), 0.0, h);
        return;
    }

    if (s.z >= 1.0) { f = s; return; }   // dead, waiting for the ring

    float h = s.w;
    float spd = 1.0 + u_simC.w * (h - 0.5) * 2.0;    // fur: per-particle speed spread
    vec2 uv = s.xy;
    vec2 v1 = texture(u_vel, uv).xy;
    vec2 mid = uv + 0.5 * dt * spd * pvel_to_uv(v1);
    vec2 v2 = texture(u_vel, mid).xy;
    uv += dt * spd * pvel_to_uv(v2);
    uv += pvel_to_uv(gauss2(uidx ^ 0xA5A5u, step_)) * u_simD.w * sqrt(dt);

    float life = u_simE.w * (0.6 + 0.8 * fract(h * 7.31));
    float age = s.z + dt / life;
    float m = u_simG.z;
    if (uv.x < -m || uv.x > 1.0 + m || uv.y < -m || uv.y > 1.0 + m) age = 2.0;
    f = vec4(uv, age, h);
}
