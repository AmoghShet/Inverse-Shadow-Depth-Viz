#version 410 core
// D2b: 2x box downsample (4 bilinear taps at half-texel offsets).
uniform sampler2D u_src;
in vec2 v_uv;
out vec4 f;
void main() {
    vec2 tx = 0.5 / vec2(textureSize(u_src, 0));
    f = 0.25 * (texture(u_src, v_uv + vec2(-tx.x, -tx.y)) + texture(u_src, v_uv + vec2(tx.x, -tx.y))
              + texture(u_src, v_uv + vec2(-tx.x, tx.y)) + texture(u_src, v_uv + vec2(tx.x, tx.y)));
}
