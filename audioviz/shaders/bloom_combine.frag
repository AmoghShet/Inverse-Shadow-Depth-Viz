#version 410 core
// P4: sum the three blurred levels (1/4, 1/8, 1/16) at 1/4 resolution.
uniform sampler2D u_l0;
uniform sampler2D u_l1;
uniform sampler2D u_l2;
uniform vec3 u_weights;
in vec2 v_uv;
out vec4 f;
void main() {
    f = texture(u_l0, v_uv) * u_weights.x + texture(u_l1, v_uv) * u_weights.y + texture(u_l2, v_uv) * u_weights.z;
}
