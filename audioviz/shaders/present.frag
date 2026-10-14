#version 410 core
// P7: present. View 0 = ASCII frame + glyph glow; 1 = raw density; 2 = flow (hue = direction,
// brightness = speed, with arrows). Soft shoulder for values above 1, then gamma encode.
#include "common.glsl"

uniform sampler2D u_frame;   // RGBA16F linear ASCII frame
uniform sampler2D u_glow;    // RGBA16F glyph glow (1/4 res)
uniform sampler2D u_dens;
uniform sampler2D u_bloom;
uniform sampler2D u_vel;
uniform int u_view;
uniform float u_glowGain;
uniform float u_flash;     // --click-test: beat pulse flash

in vec2 v_uv;
out vec4 f;

vec3 shoulder(vec3 c) {
    c = max(c, 0.0);
    vec3 hi = 0.8 + 0.2 * (1.0 - exp(-(c - 0.8) / 0.2));
    return mix(c, hi, step(0.8, c));
}

vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

void main() {
    vec3 col;
    if (u_view == 1) {
        float d = texture(u_dens, v_uv).r;
        float bl = texture(u_bloom, v_uv).r * u_lookD.x;
        float r = clamp(tone(d), 0.0, 1.0);
        col = palette(r) * (1.0 + u_look.w * r * r) + u_lookF.rgb * bl;
    } else if (u_view == 2) {
        vec2 v = texture(u_vel, v_uv).xy;
        float ang = atan(v.y, v.x);
        float mag = length(v);
        col = hsv2rgb(vec3(ang / (2.0 * PI) + 0.5, 0.7, clamp(mag / 0.25, 0.0, 1.0))) * 0.6;
        // arrows: one per 32 px block, pointing along the flow at the block centre
        vec2 px = gl_FragCoord.xy;
        vec2 blk = floor(px / 32.0) * 32.0 + 16.0;
        vec2 vb = texture(u_vel, blk / u_res.xy).xy;
        float L = clamp(length(vb) / 0.25, 0.0, 1.0) * 14.0;
        vec2 dir = length(vb) > 1e-5 ? normalize(vb) : vec2(1.0, 0.0);
        vec2 q = px - blk;
        float along = dot(q, dir);
        float across = abs(dot(q, vec2(-dir.y, dir.x)));
        float shaft = step(0.0, along) * step(along, L) * step(across, 0.8);
        float head = step(L - 4.0, along) * step(along, L) * step(across, (L - along) * 0.6 + 0.4);
        col += vec3(max(shaft, head));
        float d = texture(u_dens, v_uv).r;
        col += vec3(0.25) * clamp(tone(d), 0.0, 1.0);
    } else {
        col = texture(u_frame, v_uv).rgb + u_glowGain * u_lookF.rgb * texture(u_glow, v_uv).rgb;
    }
    col += vec3(0.35 * u_flash);
    col = shoulder(col);
    f = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);
}
