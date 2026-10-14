#version 410 core
// P2: one GL_POINT per state texel, positioned from gl_VertexID + texelFetch (no VBO).
#include "common.glsl"

uniform sampler2D u_state;
#ifdef DUMMY_ATTRIB
in float a_dummy;   // some Apple drivers draw nothing from an empty VAO
#endif

out float v_w;

void main() {
    int S = textureSize(u_state, 0).x;
    ivec2 ij = ivec2(gl_VertexID % S, gl_VertexID / S);
    vec4 s = texelFetch(u_state, ij, 0);
    float age = s.z;
    if (age >= 1.0) {
        gl_Position = vec4(-4.0, -4.0, 0.0, 1.0);
        gl_PointSize = 1.0;
        v_w = 0.0;
        return;
    }
    float w = smoothstep(0.0, 0.08, age) * (1.0 - smoothstep(0.7, 1.0, age));
    float ink = 0.45 + 1.1 * fract(s.w * 13.17);    // ink concentration varies per particle
    v_w = w * ink;
    gl_Position = vec4(s.xy * 2.0 - 1.0, 0.0, 1.0);
#ifdef DUMMY_ATTRIB
    v_w *= 1.0 + a_dummy;   // the buffer holds zeros; a real use the compiler cannot fold away
#endif
    gl_PointSize = u_simA.w;
}
