// Colouring / display pass.
//
// For every output pixel the best available source is chosen by "quality"
// (effective sample density relative to the screen, 1 = one exact sample per pixel):
//   * exact samples of the current field                quality uFieldQ
//   * the last completed image, reprojected             quality uPrevQ
//   * bilinear upsampling of coarse progressive levels  quality uFieldQ / 4^L
// Using the last *completed* image (instead of the previous frame) avoids the
// cumulative blur of resampling an already resampled image every frame.

layout(std430, binding = 1) readonly buffer ResultBuf { uvec4 res[]; };   // 2 per sample

uniform uvec2 uField;        // field size (samples)
uniform vec4  uVP;           // viewport x, y, w, h (framebuffer pixels)
uniform float uScale;        // field samples per output pixel
uniform ivec2 uSampleOrigin; // sample offset of this output region inside the field
uniform float uFieldQ;
uniform sampler2D uPalette;
uniform sampler2D uPrev;
uniform int   uHavePrev;
uniform float uPrevQ;
uniform vec2  uPrevA;        // similarity: s' = s0 + delta + A*(s - s0)
uniform vec2  uPrevD;
uniform vec2  uPrevSize;     // previous image size (pixels)
uniform vec2  uOutHalf;      // s0 (in output pixels, of the full image)
uniform vec2  uOutOffset;    // offset of viewport inside the full image (pixels, y down)
uniform vec3  uBackground;
uniform int   uFieldXf;      // 1: the field was computed for another view (still in flight while moving)
uniform vec2  uFieldA;       //    similarity screen -> field view, like uPrevA / uPrevD
uniform vec2  uFieldD;

uniform vec3  uInterior;
uniform int   uMethod;       // 0 smooth iterations, 1 distance estimate
uniform float uDensity;
uniform float uDEDensity;
uniform float uOffset;
uniform int   uMapping;      // 0 linear, 1 log, 2 sqrt, 3 auto (histogram)
uniform uint  uIterBase;
uniform uint  uMaxIterShow;
uniform float uDEStrength;
uniform float uDEScale;
uniform float uSlopeStrength;
uniform vec2  uLightDir;
uniform float uLightHeight;
uniform int   uShowFallback;
uniform sampler2D uAutoTex;  // cumulative distribution of log2(mu), 1024 x 1
uniform vec2  uAutoRange;    // log2(mu) range covered by uAutoTex
uniform float uAutoCycles;
uniform float uGlow;         // brighten near the set (DE based)
uniform vec3  uGlowColor;
uniform float uStripes;      // stripe-average texture amount
uniform int   uPattern;      // 0 none, 1 binary decomposition, 2 field lines, 3 grid
uniform float uPatternStrength;
uniform float uPatternFreq;

uniform int   uConsistent;   // 1: a field's level-0 samples only where the 2-px lattice around is
                             //    complete (no dot grid over the older picture while refining);
                             //    coarse levels interpolated with Catmull-Rom.  0: v3.6 behaviour
uniform int   uFieldDone;    // the field is complete (skip the completeness tests)

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 srcOut;   // diagnostics: which source the pixel shows (motion study)

// sources (srcOut.r * 15): 1 exact, 2 exact reprojected, 3 older picture, 4-6 coarse level 1-3,
// 7 partial blend, 8 nothing (background)
void emit(vec3 c, float q, int src) {
    fragColor = vec4(c, q);
    srcOut = vec4(float(src) / 15.0, clamp(q, 0.0, 1.0), 0.0, 1.0);
}

const uint RES_INTERIOR = 0xFFFFFFFEu;
const float TAU = 6.283185307179586;

vec3 colorize(uvec4 a, uvec4 b) {
    if (a.x >= RES_INTERIOR || a.x > uMaxIterShow) return uInterior;
    float frac = uintBitsToFloat(a.y);
    float de = uintBitsToFloat(a.z);
    float slope = uintBitsToFloat(a.w);
    float zarg = uintBitsToFloat(b.x);
    float stripe = uintBitsToFloat(b.y);
    float t;
    if (uMethod == 1 && de > 0.0) {
        t = -log2(max(de * uDEScale, 1e-9)) * uDEDensity;
    } else {
        float mu = max(float(int(a.x) - int(uIterBase)) + frac, 0.0);
        if (uMapping == 1) t = log2(mu + 1.0) * uDensity * 8.0;
        else if (uMapping == 2) t = sqrt(mu) * uDensity * 4.0;
        else if (uMapping == 3) {
            float x = (log2(float(a.x) + frac) - uAutoRange.x) / max(uAutoRange.y - uAutoRange.x, 1e-6);
            t = texture(uAutoTex, vec2(clamp(x, 0.0, 1.0) * (1023.0 / 1024.0) + 0.5 / 1024.0, 0.5)).r * uAutoCycles;
        }
        else t = mu * uDensity;
    }
    t += uStripes * stripe;
    t += uOffset;
    vec3 col = texture(uPalette, vec2(fract(t), 0.5)).rgb;
    if (uStripes > 0.0) col *= mix(1.0, 0.75 + 0.5 * stripe, min(uStripes, 1.0) * 0.5);
    if (uPattern == 1) {
        if (sin(zarg) < 0.0) col *= 1.0 - 0.6 * uPatternStrength;
    } else if (uPattern >= 2) {
        float g = abs(fract(zarg * uPatternFreq / TAU + 0.5) - 0.5) * 2.0;     // 0 on a field line
        float line = 1.0 - smoothstep(0.0, 0.12, g);
        if (uPattern == 3) {
            float r = 1.0 - abs(fract(frac) - 0.5) * 2.0;                      // 0 on a band edge
            line = max(line, 1.0 - smoothstep(0.0, 0.08, r));
        }
        col *= 1.0 - uPatternStrength * line;
    }
    if (de > 0.0) {
        float f = clamp(de * uDEScale, 0.0, 1.0);
        if (uDEStrength > 0.0) col *= mix(1.0, sqrt(f), uDEStrength);
        if (uGlow > 0.0) col = mix(col, uGlowColor, uGlow * (1.0 - f) * (1.0 - f));
    }
    if (uSlopeStrength > 0.0 && de > 0.0) {
        vec2 u = vec2(cos(slope), sin(slope));
        float sh = (dot(u, uLightDir) + uLightHeight) / (1.0 + uLightHeight);
        sh = clamp(sh, 0.0, 1.0);
        col *= mix(1.0, 0.25 + 0.95 * sh, uSlopeStrength);
    }
    return col;
}

bool fetchExact(ivec2 s, out vec3 c) {
    s = clamp(s, ivec2(0), ivec2(uField) - 1);
    uint i = uint(s.y) * uField.x + uint(s.x);
    uvec4 a = res[2u * i];
    c = vec3(0.0);
    if (a.x == 0u) return false;
    c = colorize(a, res[2u * i + 1u]);
    return true;
}

bool doneAt(ivec2 s) {
    s = clamp(s, ivec2(0), ivec2(uField) - 1);
    return res[2u * (uint(s.y) * uField.x + uint(s.x))].x != 0u;
}

// Are the four level-L lattice samples around field position f finished?  (no colouring)
bool latticeDone(vec2 f, int L) {
    int st = 1 << L;
    ivec2 base = ivec2(floor(f / float(st))) * st;
    ivec2 lim = ivec2(uField) - 1;
    ivec2 b3 = base + ivec2(st);
    if (b3.x > lim.x) b3.x = base.x;
    if (b3.y > lim.y) b3.y = base.y;
    return doneAt(base) && doneAt(ivec2(b3.x, base.y)) && doneAt(ivec2(base.x, b3.y)) && doneAt(b3);
}

// bilinear interpolation of the level-L grid (samples at multiples of 2^L)
bool fetchCoarse(vec2 f, int L, out vec3 c) {
    float st = float(1 << L);
    vec2 g = f / st;
    vec2 gf = floor(g);
    vec2 t = g - gf;
    ivec2 base = ivec2(gf) * (1 << L);
    vec3 c00, c10, c01, c11;
    ivec2 d = ivec2(1 << L, 0);
    ivec2 e = ivec2(0, 1 << L);
    ivec2 lim = ivec2(uField) - 1;
    ivec2 b0 = base;
    ivec2 b1 = base + d;
    ivec2 b2 = base + e;
    ivec2 b3 = base + d + e;
    if (b3.x > lim.x) { b1.x = b0.x; b3.x = b0.x; }
    if (b3.y > lim.y) { b2.y = b0.y; b3.y = b0.y; }
    if (!fetchExact(b0, c00) || !fetchExact(b1, c10) || !fetchExact(b2, c01) || !fetchExact(b3, c11)) {
        c = vec3(0.0);
        return false;
    }
    c = mix(mix(c00, c10, t.x), mix(c01, c11, t.x), t.y);
    return true;
}

// Catmull-Rom weights: sharper than bilinear when a picture is resampled (rotation, zoom),
// clamped to the four nearest values so edges do not ring.
vec4 crWeights(float t) {
    float t2 = t * t, t3 = t2 * t;
    return vec4(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0,
                -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2);
}

// Catmull-Rom over the 4x4 level-L lattice points around f: smooth curves instead of the
// straight-edged "tents" bilinear interpolation draws through a sparse lattice (they look
// like a grid); falls back to bilinear when a lattice point is missing
bool fetchCoarseSmooth(vec2 f, int L, out vec3 c) {
    int st = 1 << L;
    vec2 g = f / float(st);
    ivec2 i0 = ivec2(floor(g));
    vec2 t = g - vec2(i0);
    ivec2 gmax = (ivec2(uField) - 1) / st;
    vec4 wx = crWeights(t.x), wy = crWeights(t.y);
    vec3 sum = vec3(0.0), mn = vec3(1e9), mx = vec3(-1e9);
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i) {
            ivec2 gp = clamp(i0 + ivec2(i - 1, j - 1), ivec2(0), gmax);
            vec3 cc;
            if (!fetchExact(gp * st, cc)) return fetchCoarse(f, L, c);
            sum += cc * (wx[i] * wy[j]);
            if (i == 1 || i == 2) if (j == 1 || j == 2) { mn = min(mn, cc); mx = max(mx, cc); }
        }
    c = clamp(sum, mn, mx);
    return true;
}

vec4 sampleSharp(sampler2D tex, vec2 uv) {
    ivec2 sz = textureSize(tex, 0);
    vec2 p = uv * vec2(sz) - 0.5;
    vec2 fl = floor(p);
    vec2 t = p - fl;
    ivec2 b = ivec2(fl) - 1;
    vec4 wx = crWeights(t.x), wy = crWeights(t.y);
    vec3 sum = vec3(0.0), mn = vec3(1e9), mx = vec3(-1e9);
    float amin = 1e9;
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i) {
            vec4 c = texelFetch(tex, clamp(b + ivec2(i, j), ivec2(0), sz - 1), 0);
            sum += c.rgb * (wx[i] * wy[j]);
            if (i == 1 || i == 2) if (j == 1 || j == 2) {
                mn = min(mn, c.rgb); mx = max(mx, c.rgb); amin = min(amin, c.a);
            }
        }
    vec4 bil = texture(tex, uv);
    if (amin <= 0.0) return bil;                 // next to an empty part: plain bilinear
    return vec4(clamp(sum, mn, mx), bil.a);
}

// The snapshot keeps its per-pixel quality in alpha; uPrevQ is the factor its reprojection
// onto the screen applies (magnification, rotation).
bool prevColor(vec2 sp, out vec3 c, out float pq) {
    c = uBackground;
    pq = -1.0;
    if (uHavePrev == 0) return false;
    vec2 d = sp - uOutHalf;
    vec2 q = uOutHalf + uPrevD + vec2(uPrevA.x * d.x - uPrevA.y * d.y, uPrevA.y * d.x + uPrevA.x * d.y);
    vec2 uv = vec2(q.x / uPrevSize.x, 1.0 - q.y / uPrevSize.y);
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return false;
    vec4 t = sampleSharp(uPrev, uv);
    if (t.a <= 0.0) return false;
    c = t.rgb;
    pq = t.a * uPrevQ;
    return true;
}

void main() {
    vec2 lp = gl_FragCoord.xy - uVP.xy;
    float lx = lp.x - 0.5;
    float ly = uVP.w - 1.0 - (lp.y - 0.5);          // y down, local to viewport
    vec2 sp = vec2(lx + 0.5, ly + 0.5) + uOutOffset; // screen position in full image

    // ---- exact samples
    vec3 exact = vec3(0.0);
    float have = 0.0;               // fraction of the pixel covered by finished samples
    vec2 fpos;                      // position in field sample coordinates
    bool inField = true;
    if (uFieldXf != 0) {
        // reprojected field: bilinear interpolation of finished samples
        vec2 d = sp - uOutHalf;
        vec2 q = uOutHalf + uFieldD + vec2(uFieldA.x * d.x - uFieldA.y * d.y, uFieldA.y * d.x + uFieldA.x * d.y);
        vec2 f = q * uScale - 0.5 + vec2(uSampleOrigin);
        fpos = f;
        inField = f.x >= -0.5 && f.y >= -0.5 && f.x <= float(uField.x) - 0.5 && f.y <= float(uField.y) - 0.5;
        if (inField) {
            ivec2 i0 = ivec2(floor(f));
            vec2 t = f - vec2(i0);
            // Catmull-Rom over the 4x4 neighbourhood when it is complete (sharp), else bilinear
            vec4 wx = crWeights(t.x), wy = crWeights(t.y);
            vec3 sum = vec3(0.0), mn = vec3(1e9), mx = vec3(-1e9);
            bool all16 = true;
            for (int j = 0; j < 4 && all16; ++j)
                for (int i = 0; i < 4; ++i) {
                    vec3 c;
                    if (!fetchExact(i0 + ivec2(i - 1, j - 1), c)) { all16 = false; break; }
                    sum += c * (wx[i] * wy[j]);
                    if (i == 1 || i == 2) if (j == 1 || j == 2) { mn = min(mn, c); mx = max(mx, c); }
                }
            if (all16) {
                exact = clamp(sum, mn, mx);
                have = 1.0;
            } else {
                for (int j = 0; j < 2; ++j)
                    for (int i = 0; i < 2; ++i) {
                        vec3 c;
                        float w = (i == 0 ? 1.0 - t.x : t.x) * (j == 0 ? 1.0 - t.y : t.y);
                        if (fetchExact(i0 + ivec2(i, j), c)) { exact += c * w; have += w; }
                    }
            }
        }
    } else if (uScale >= 1.0) {
        int ss = int(uScale + 0.5);
        ivec2 base = ivec2(int(lx) * ss, int(ly) * ss) + uSampleOrigin;
        float w = 1.0 / float(ss * ss);
        for (int j = 0; j < ss; ++j)
            for (int i = 0; i < ss; ++i) {
                vec3 c;
                if (fetchExact(base + ivec2(i, j), c)) { exact += c * w; have += w; }
            }
        fpos = vec2(base) + 0.5 * float(ss - 1);
    } else {
        vec2 f = (vec2(lx, ly) + 0.5) * uScale - 0.5 + vec2(uSampleOrigin);
        fpos = f;
        ivec2 i0 = ivec2(floor(f));
        vec2 t = f - vec2(i0);
        for (int j = 0; j < 2; ++j)
            for (int i = 0; i < 2; ++i) {
                vec3 c;
                float w = (i == 0 ? 1.0 - t.x : t.x) * (j == 0 ? 1.0 - t.y : t.y);
                if (fetchExact(i0 + ivec2(i, j), c)) { exact += c * w; have += w; }
            }
    }
    int exSrc = uFieldXf != 0 ? 2 : 1;
    // While a field refines, its finished samples form sparse lattices (every 8th, 4th, 2nd
    // sample, then the rows of the last level).  Shown sharp between pixels that come from
    // another source (the older picture) they form a dot grid or stripes, so a finished sample
    // counts only where at least 3 of its 4 neighbours are finished too (the last sweep has
    // passed; a single sample still running, next to the set, does not count against it).
    // Before that the field takes part through its coarse levels, which are smooth and agree
    // with the samples at the lattice points.
    bool dense = uConsistent == 0 || uFieldDone != 0 || (uFieldXf == 0 && uScale >= 1.5);
    if (!dense && inField) {
        ivec2 n0 = ivec2(floor(fpos + 0.5));
        int nb = int(doneAt(n0 + ivec2(1, 0))) + int(doneAt(n0 - ivec2(1, 0)))
               + int(doneAt(n0 + ivec2(0, 1))) + int(doneAt(n0 - ivec2(0, 1)));
        dense = nb >= 3 && doneAt(n0);
    }
    bool exactOk = have > 0.999 && dense;
    if (exactOk && uFieldQ >= uPrevQ) { emit(exact / have, uFieldQ, exSrc); return; }

    // ---- previous completed image
    vec3 pc;
    float pq;
    bool prevOk = prevColor(sp, pc, pq);
    float bestQ = prevOk ? pq : -1.0;
    vec3 best = pc;
    int src = prevOk ? 3 : 8;
    if (exactOk && uFieldQ > bestQ) { bestQ = uFieldQ; best = exact / have; src = exSrc; }

    // ---- coarse progressive levels (smooth)
    if (uShowFallback != 0 && inField) {
        for (int L = 1; L <= 3; ++L) {
            float q = uFieldQ / float(1 << (2 * L));
            if (q <= bestQ) break;
            vec3 c;
            bool ok = uConsistent != 0 ? fetchCoarseSmooth(fpos, L, c) : fetchCoarse(fpos, L, c);
            if (ok) { bestQ = q; best = c; src = 3 + L; break; }
        }
    }
    if (bestQ < 0.0) {
        best = (have > 0.0) ? exact / have : uBackground;
        bestQ = (have > 0.0) ? 0.001 : 0.0;
        src = (have > 0.0) ? 7 : 8;
    } else if (have > 0.0 && !exactOk && bestQ < uFieldQ && dense) {
        // partially finished supersampled pixel: blend what we have
        best = mix(best, exact / have, have);
        src = 7;
    }
    emit(best, bestQ, src);   // alpha: quality (kept by snapshots)
}
