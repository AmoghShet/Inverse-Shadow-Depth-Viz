#version 410 core
// HUD drawn as glyphs: a small text grid (R = char code, G = colour index) over a dim backdrop.
uniform sampler2D u_text;      // RG8 normalised, top-origin rows
uniform sampler2D u_hudAtlas;  // R8, printable ASCII 32..126 in one row
uniform vec2 u_hudCell;
uniform vec2 u_hudOrigin;
uniform vec2 u_outSize;

out vec4 f;

void main() {
    vec2 fc = vec2(gl_FragCoord.x, u_outSize.y - gl_FragCoord.y) - u_hudOrigin;
    ivec2 tsz = textureSize(u_text, 0);
    if (fc.x < 0.0 || fc.y < 0.0) discard;
    ivec2 cid = ivec2(floor(fc / u_hudCell));
    if (cid.x >= tsz.x || cid.y >= tsz.y) discard;
    vec2 t = texelFetch(u_text, cid, 0).rg;
    int code = int(t.r * 255.0 + 0.5);
    int ci = int(t.g * 255.0 + 0.5);
    if (code == 0) discard;            // outside any text line
    ivec2 inC = ivec2(fc) - cid * ivec2(u_hudCell);
    float ink = 0.0;
    if (code > 32 && code < 127)
        ink = texelFetch(u_hudAtlas, ivec2((code - 32) * int(u_hudCell.x) + inC.x, inC.y), 0).r;
    vec3 c = ci == 1 ? vec3(1.0, 0.4, 0.35) : ci == 2 ? vec3(0.55, 0.93, 1.0) : vec3(0.78, 0.83, 0.88);
    // premultiplied: backdrop alpha 0.55, glyph on top
    float a = max(0.55, ink);
    f = vec4(c * ink, a);
}
