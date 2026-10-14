#version 410 core
// Camera ghosts. R = the live camera level, G = a peak-hold of it that fades (half-life
// camE.x) and drifts gently with the flow. trails.frag pulls the density toward
// max(R, camE.z * G): the live image stays bright and clear, and anything that moves leaves a
// dimmer, gray trail that fades out.
#include "common.glsl"

uniform sampler2D u_ghost;   // previous ghost image (screen-space uv)
uniform sampler2D u_cam;     // camera frame (R8, mipmapped)
uniform sampler2D u_vel;

in vec2 v_uv;
out vec4 f;

void main() {
    float dt = u_time.y;
    vec2 v = texture(u_vel, v_uv).xy;
    vec2 back = v_uv - pvel_to_uv(v) * dt * u_camE.y;
    float g = texture(u_ghost, back).g * pow(0.5, dt / max(u_camE.x, 0.01));
    float c = cam_level(u_cam, v_uv);
    f = vec4(c, max(g, c), 0.0, 1.0);
}
