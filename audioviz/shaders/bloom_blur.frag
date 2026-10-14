#version 410 core
// P4: separable 9-tap gaussian using 5 linear taps.
uniform sampler2D u_src;
uniform vec2 u_dir;   // (1,0) or (0,1)
in vec2 v_uv;
out vec4 f;
void main() {
    vec2 tx = u_dir / vec2(textureSize(u_src, 0));
    vec4 c = texture(u_src, v_uv) * 0.2270270270;
    c += texture(u_src, v_uv + tx * 1.3846153846) * 0.3162162162;
    c += texture(u_src, v_uv - tx * 1.3846153846) * 0.3162162162;
    c += texture(u_src, v_uv + tx * 3.2307692308) * 0.0702702703;
    c += texture(u_src, v_uv - tx * 3.2307692308) * 0.0702702703;
    f = c;
}
