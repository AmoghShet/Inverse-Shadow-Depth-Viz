#version 410 core
// P3: advect the previous density along the flow (semi-Lagrangian), decay, subtract a small bias.
// Runs before the splat so fresh particles stay crisp on top of the soft history.
#include "common.glsl"

uniform sampler2D u_dens;
uniform sampler2D u_vel;
uniform sampler2D u_ghost;  // R live camera level, G fading hold (ghost.frag)

in vec2 v_uv;
out vec4 f;

void main() {
    float dt = u_time.y;
    float steps60 = dt * 60.0;
    vec2 v = texture(u_vel, v_uv).xy;
    vec2 back = v_uv - pvel_to_uv(v) * dt * u_simA.z;
    float d = texture(u_dens, back).r;
    d = d * pow(u_simA.y, steps60) - u_simG.x * steps60;
    if (u_camA.w > 0.5 && u_camD.x > 0.0) {
        // Camera as smoke: pull the density up toward the camera image (never down, so particle
        // smoke is untouched). The image is sampled upstream along the flow (it wobbles with the
        // smoke) and broken up by drifting noise; whatever the flow carries away decays like any
        // trail. One field, one tone curve: the brightest parts of the image are the brightest
        // smoke. relax sets clarity: high follows the camera closely, low smears it.
        vec2 warp = pvel_to_uv(v) * u_camD.w;
        vec2 gh = texture(u_ghost, v_uv - warp).rg;
        float c = max(gh.r, u_camE.z * gh.g);   // live image bright, trails gray
        vec2 pp = uv_to_p(v_uv);
        float n = 0.65 * noise3(vec3(pp * 7.0, u_time.x * 0.5)) + 0.35 * noise3(vec3(pp * 19.0 + 3.1, u_time.x * 1.1));
        c *= mix(1.0, smoothstep(-0.25, 0.45, n), u_camD.z);
        float target = c * u_camD.x;
        float a = 1.0 - pow(1.0 - u_camD.y, steps60);
        if (target > d) d += (target - d) * a;
    }
    f = vec4(max(d, 0.0));
}
