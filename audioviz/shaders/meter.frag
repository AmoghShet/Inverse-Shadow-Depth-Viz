#version 410 core
// Exposure meter: each output texel averages a block of the density buffer.
// r = mean density, g = covered fraction (d > 0.01), b = block max.
uniform sampler2D u_dens;
in vec2 v_uv;
out vec4 f;
void main() {
    ivec2 osz = ivec2(64, 36);
    vec2 cell = 1.0 / vec2(osz);
    vec2 base = floor(v_uv * vec2(osz)) * cell;
    float sum = 0.0, cov = 0.0, mx = 0.0;
    for (int j = 0; j < 6; ++j)
        for (int i = 0; i < 6; ++i) {
            float d = texture(u_dens, base + (vec2(i, j) + 0.5) / 6.0 * cell).r;
            sum += d;
            cov += step(0.01, d);
            mx = max(mx, d);
        }
    f = vec4(sum / 36.0, cov / 36.0, mx, 1.0);
}
