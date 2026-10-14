#version 410 core
// D2b: separable 9-tap gaussian with 5 linear taps; u_spread widens it (harmonic -> halation radius).
uniform sampler2D u_src;
uniform vec2 u_dir;
uniform float u_spread;
in vec2 v_uv;
out vec4 f;
void main() {
    vec2 tx = u_dir * max(u_spread, 0.25) / vec2(textureSize(u_src, 0));
    vec4 c = texture(u_src, v_uv) * 0.2270270270;
    c += texture(u_src, v_uv + tx * 1.3846153846) * 0.3162162162;
    c += texture(u_src, v_uv - tx * 1.3846153846) * 0.3162162162;
    c += texture(u_src, v_uv + tx * 3.2307692308) * 0.0702702703;
    c += texture(u_src, v_uv - tx * 3.2307692308) * 0.0702702703;
    f = c;
}
