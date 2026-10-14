#version 410 core
// P0: flow field. 3-octave curl noise + drift + divergence-free emitter jets (kicks boost them)
// + external velocity.
// Output: velocity in p-space units per second (RG16F, 1/4 res).
#include "common.glsl"

uniform sampler2D u_ext;   // external velocity (camera stir), p-space units/s; 1x1 zero until M7
uniform vec2 u_extScale;   // uv -> ext uv scale (cover-crop), centred

in vec2 v_uv;
out vec4 f;

void main() {
    vec2 p = uv_to_p(v_uv);
    float t = u_time.x;
    mat2 R = mat2(u_simF.z, u_simF.w, -u_simF.w, u_simF.z);   // seed rotation of the noise domain
    vec2 drift = u_simE.xy;
    vec2 q = R * (p - drift * t * 0.6) + u_simF.xy;

    // Octave 1 crossfades to a finer copy (sub bass) instead of rescaling the domain, which
    // would slide the whole pattern.
    vec2 vq = u_simC.x * curl(q * u_simB.x, t * u_simD.x)
            + u_simH.y * curl(q * u_simH.x + vec2(-41.2, 3.3), t * u_simD.x + 13.0)
            + u_simC.y * curl(q * u_simB.y + vec2(17.3, -9.1), t * u_simD.y + 31.0)
            + u_simC.z * curl(q * u_simB.z + vec2(-5.7, 23.9), t * u_simD.z + 67.0);
    vec2 v = transpose(R) * vq * u_simB.w + drift;

    int n = int(u_emitB.x);
    for (int i = 0; i < 8; ++i) {
        if (i >= n) break;
        vec2 d = p - u_emitters[i].xy;
        float r2 = dot(d, d);
        // Jet from a stream function psi = J*s*tanh(b/s)*exp(-a^2/2L^2) in jet coordinates
        // (a along, b across): divergence-free, so particles never pile up at the jet's front, and
        // the return flow curls the plume into a mushroom-cap vortex pair like real ink.
        vec2 jd = u_emittersB[i].xy;
        // a kick adds to the jet for ~150 ms: the puff leaves as a mushroom cap. (A radial source
        // impulse is compressive and, repeated on every kick, packed particles into a solid ring.)
        float J = u_emittersB[i].z + u_emittersB[i].w;
        if (J > 0.0) {
            float s = max(u_emitters[i].w * 3.0, 0.03);
            float L = s * 3.0 * (1.0 + 0.5 * u_emittersB[i].w / (u_emittersB[i].w + 0.1));
            float a = dot(d, jd) - 0.8 * L;
            float b = dot(d, vec2(-jd.y, jd.x));
            float th = tanh(b / s);
            float g = exp(-a * a / (2.0 * L * L));
            float va = J * (1.0 - th * th) * g;
            float vb = J * s * th * (a / (L * L)) * g;
            v += jd * va + vec2(-jd.y, jd.x) * vb;
        }
    }

    vec2 euv = (v_uv - 0.5) * u_extScale + 0.5;
    v += u_lookE.y * texture(u_ext, euv).xy;
    f = vec4(v * u_simA.x, 0.0, 1.0);
}
