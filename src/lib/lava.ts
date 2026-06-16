/**
 * Inductive ML — molten crimson-silk lava shader with a Blender-style bloom
 * pipeline plus drifting ember sparks.
 *
 * Ported from the design prototype's `lava.js` into a typed module. Two
 * additions over the prototype, both for runtime performance:
 *   - the render loop is fully paused when the canvas is off-screen
 *     (IntersectionObserver) or the tab is hidden (Page Visibility API);
 *   - `prefers-reduced-motion` renders a single static frame and never loops.
 *
 * No globals are attached to `window`; callers import `initLava` / `initEmbers`
 * directly and pass compile-time-typed option literals.
 */

const VERT =
  'attribute vec2 a; varying vec2 vUv; void main(){ vUv = a*0.5+0.5; gl_Position = vec4(a,0.0,1.0); }';

const SCENE_FS = [
  'precision highp float;',
  'varying vec2 vUv;',
  'uniform vec2 u_res; uniform float u_time; uniform float u_dim; uniform float u_scale;',
  'float hash(vec2 p){ return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453123); }',
  'float vnoise(vec2 p){ vec2 i=floor(p), f=fract(p); vec2 u=f*f*(3.0-2.0*f);',
  '  float a=hash(i), b=hash(i+vec2(1.0,0.0)), c=hash(i+vec2(0.0,1.0)), d=hash(i+vec2(1.0,1.0));',
  '  return mix(mix(a,b,u.x), mix(c,d,u.x), u.y); }',
  'float fbm(vec2 p){ float v=0.0, a=0.5; mat2 m=mat2(1.6,1.2,-1.2,1.6);',
  '  for(int i=0;i<5;i++){ v+=a*vnoise(p); p=m*p; a*=0.5; } return v; }',
  'vec3 lava(float t){ t=clamp(t,0.0,1.0);',
  '  vec3 c1=vec3(0.15,0.008,0.03);',
  '  vec3 c2=vec3(0.30,0.013,0.05);',
  '  vec3 c3=vec3(0.50,0.025,0.06);',
  '  vec3 c4=vec3(0.70,0.045,0.07);',
  '  vec3 c5=vec3(0.86,0.085,0.09);',
  '  vec3 c6=vec3(0.98,0.13,0.11);',
  '  vec3 col=mix(c1,c2,smoothstep(0.0,0.22,t));',
  '  col=mix(col,c3,smoothstep(0.22,0.42,t));',
  '  col=mix(col,c4,smoothstep(0.42,0.6,t));',
  '  col=mix(col,c5,smoothstep(0.6,0.8,t));',
  '  col=mix(col,c6,smoothstep(0.85,1.0,t));',
  '  return col; }',
  'void main(){',
  '  float aspect = u_res.x/u_res.y;',
  '  vec2 uv = vUv;',
  '  vec2 p = vUv-0.5; p.x *= aspect; p *= u_scale;',
  '  float t = u_time*0.075;',
  '  vec2 q = vec2(fbm(p + t), fbm(p+vec2(5.2,1.3)-t));',
  '  vec2 r = vec2(fbm(p+1.8*q+vec2(1.7,9.2)+0.15*t), fbm(p+1.8*q+vec2(8.3,2.8)+0.126*t));',
  '  float f = fbm(p+2.0*r);',
  '  float heat = clamp((f*0.6 + r.x*0.45)*1.08, 0.0, 1.0);',
  '  vec3 col = lava(heat);',
  '  float tex = fbm(p*6.0 + 3.0*r + 0.5*t);',
  '  col *= 0.84 + 0.32*tex;',
  '  float wob = 0.30*fbm(vec2(p.x*0.5, p.y*0.32 + 0.22*t));',
  '  float fold = sin(vUv.x*aspect*u_scale*4.6 + wob*5.0 + 1.3*r.x);',
  '  float foldShade = 0.5 + 0.5*fold;',
  '  float silk = pow(max(fold,0.0), 5.0);',
  '  col *= 0.70 + 0.42*foldShade;',
  '  col += vec3(1.0,0.16,0.12)*silk*0.34;',
  '  float e=0.06; float gx=fbm(p+vec2(e,0.0)+2.0*r)-fbm(p-vec2(e,0.0)+2.0*r); float gy=fbm(p+vec2(0.0,e)+2.0*r)-fbm(p-vec2(0.0,e)+2.0*r);',
  '  vec3 nrm=normalize(vec3(-gx,-gy,0.22));',
  '  vec3 L=normalize(vec3(0.45*sin(u_time*0.05)+0.2, 0.55, 0.8));',
  '  float spec=pow(max(dot(nrm,L),0.0), 38.0);',
  '  col += vec3(1.0,0.15,0.11)*spec*0.7;',
  '  float sheen = smoothstep(0.42,0.92,f + 0.28*r.y);',
  '  col += vec3(1.0,0.17,0.12)*sheen*0.26;',
  '  float ridge = smoothstep(0.8,1.0, fbm(p*3.0 + 4.0*r + t));',
  '  col += vec3(1.0,0.17,0.12)*ridge*0.40;',
  '  vec2 lp = vec2(0.5+0.3*sin(u_time*0.05), 0.4+0.22*cos(u_time*0.045));',
  '  col += vec3(1.0,0.68,0.36)*smoothstep(0.6,0.0,distance(uv,lp))*0.06;',
  '  col *= mix(0.93,1.0, smoothstep(1.3,0.1,length(uv-vec2(0.5))));',
  '  gl_FragColor = vec4(col*u_dim, 1.0);',
  '}',
].join('\n');

const BRIGHT_FS = [
  'precision highp float;',
  'varying vec2 vUv;',
  'uniform sampler2D u_tex; uniform float u_threshold; uniform float u_knee;',
  'void main(){',
  '  vec3 c = texture2D(u_tex, vUv).rgb;',
  '  float l = max(c.r, max(c.g, c.b));',
  '  float k = smoothstep(u_threshold, u_threshold + u_knee, l);',
  '  gl_FragColor = vec4(c*k, 1.0);',
  '}',
].join('\n');

const BLUR_FS = [
  'precision highp float;',
  'varying vec2 vUv;',
  'uniform sampler2D u_tex; uniform vec2 u_dir; uniform vec2 u_texel; uniform float u_spread;',
  'void main(){',
  '  vec2 o1 = u_dir*u_texel*1.3846153846*u_spread;',
  '  vec2 o2 = u_dir*u_texel*3.2307692308*u_spread;',
  '  vec3 s = texture2D(u_tex, vUv).rgb*0.2270270270;',
  '  s += texture2D(u_tex, vUv+o1).rgb*0.3162162162;',
  '  s += texture2D(u_tex, vUv-o1).rgb*0.3162162162;',
  '  s += texture2D(u_tex, vUv+o2).rgb*0.0702702703;',
  '  s += texture2D(u_tex, vUv-o2).rgb*0.0702702703;',
  '  gl_FragColor = vec4(s, 1.0);',
  '}',
].join('\n');

const COMP_FS = [
  'precision highp float;',
  'varying vec2 vUv;',
  'uniform sampler2D u_scene; uniform sampler2D u_bloom;',
  'uniform float u_intensity; uniform float u_exposure;',
  'void main(){',
  '  vec3 s = texture2D(u_scene, vUv).rgb;',
  '  vec3 b = texture2D(u_bloom, vUv).rgb;',
  '  vec3 c = s + b*u_intensity;',
  '  c = vec3(1.0) - exp(-c*u_exposure);',
  '  float l = dot(c, vec3(0.299,0.587,0.114)); c = clamp(mix(vec3(l), c, 1.32), 0.0, 1.0);',
  '  gl_FragColor = vec4(c, 1.0);',
  '}',
].join('\n');

export interface LavaOptions {
  dim?: number;
  scale?: number;
  speed?: number;
  animate?: boolean;
  bloom?: number;
  threshold?: number;
  exposure?: number;
  spread?: number;
  passes?: number;
}

export interface EmberOptions {
  count?: number;
}

export interface LavaController {
  /** Stop the loop and release the WebGL resources. */
  destroy: () => void;
}

interface RenderTarget {
  fb: WebGLFramebuffer;
  tex: WebGLTexture;
  w: number;
  h: number;
  ok: boolean;
}

function prefersReducedMotion(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  );
}

/**
 * Drives a callback via requestAnimationFrame, but only while the canvas is
 * both visible (tab focused) and intersecting the viewport. When
 * `prefers-reduced-motion` is set, the callback runs exactly once and never
 * loops. Returns a cleanup function that detaches all listeners.
 */
function makeAnimator(
  canvas: HTMLElement,
  speed: number,
  staticTime: number,
  draw: (time: number) => void,
): () => void {
  const reduced = prefersReducedMotion();
  let rafId = 0;
  let lastNow = 0;
  let elapsed = staticTime;
  let visible = typeof document === 'undefined' ? true : !document.hidden;
  let intersecting = true;

  const active = (): boolean => visible && intersecting && !reduced;

  const frame = (now: number): void => {
    const dt = Math.min((now - lastNow) / 1000, 0.05);
    lastNow = now;
    elapsed += dt * speed;
    draw(elapsed);
    rafId = requestAnimationFrame(frame);
  };

  const start = (): void => {
    if (rafId !== 0 || !active()) return;
    lastNow = performance.now();
    rafId = requestAnimationFrame(frame);
  };

  const stop = (): void => {
    if (rafId === 0) return;
    cancelAnimationFrame(rafId);
    rafId = 0;
  };

  const onVisibility = (): void => {
    visible = !document.hidden;
    if (active()) start();
    else stop();
  };

  const observer =
    typeof IntersectionObserver === 'function'
      ? new IntersectionObserver(
          (entries) => {
            for (const entry of entries) intersecting = entry.isIntersecting;
            if (active()) start();
            else stop();
          },
          { threshold: 0 },
        )
      : null;

  // Paint an immediate first frame so the canvas is never blank.
  draw(elapsed);

  if (reduced) {
    // Static composition only — repaint on resize but never animate.
    return () => {
      observer?.disconnect();
    };
  }

  document.addEventListener('visibilitychange', onVisibility);
  observer?.observe(canvas);
  start();

  return () => {
    stop();
    document.removeEventListener('visibilitychange', onVisibility);
    observer?.disconnect();
  };
}

export function initLava(canvas: HTMLCanvasElement, opts: LavaOptions = {}): LavaController | null {
  const dim = opts.dim ?? 1.0;
  const scale = opts.scale ?? 2.1;
  const speed = opts.speed ?? 1.0;
  const animate = opts.animate !== false;
  const bloom = opts.bloom ?? 1.0;
  const threshold = opts.threshold ?? 0.66;
  const exposure = opts.exposure ?? 0.95;
  const spread = opts.spread ?? 1.7;
  const passes = opts.passes ?? 6;

  const ctx = canvas.getContext('webgl', { antialias: false });
  const gl = ctx instanceof WebGLRenderingContext ? ctx : null;
  if (!gl) return null;
  const glc: WebGLRenderingContext = gl;

  function compile(type: number, src: string): WebGLShader | null {
    const s = glc.createShader(type);
    if (!s) return null;
    glc.shaderSource(s, src);
    glc.compileShader(s);
    if (!glc.getShaderParameter(s, glc.COMPILE_STATUS)) {
      console.warn(glc.getShaderInfoLog(s));
    }
    return s;
  }

  function program(fs: string): WebGLProgram | null {
    const p = glc.createProgram();
    const vert = compile(glc.VERTEX_SHADER, VERT);
    const frag = compile(glc.FRAGMENT_SHADER, fs);
    if (!p || !vert || !frag) return null;
    glc.attachShader(p, vert);
    glc.attachShader(p, frag);
    glc.linkProgram(p);
    if (!glc.getProgramParameter(p, glc.LINK_STATUS)) {
      console.warn(glc.getProgramInfoLog(p));
      return null;
    }
    return p;
  }

  const pScene = program(SCENE_FS);
  const pBright = program(BRIGHT_FS);
  const pBlur = program(BLUR_FS);
  const pComp = program(COMP_FS);
  if (!pScene) return null;
  const sceneProg: WebGLProgram = pScene;

  const attribOf = new WeakMap<WebGLProgram, number>();
  function attrib(p: WebGLProgram): number {
    const cached = attribOf.get(p);
    if (cached !== undefined) return cached;
    const loc = glc.getAttribLocation(p, 'a');
    attribOf.set(p, loc);
    return loc;
  }

  const buf = glc.createBuffer();
  glc.bindBuffer(glc.ARRAY_BUFFER, buf);
  glc.bufferData(glc.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), glc.STATIC_DRAW);

  function bindQuad(p: WebGLProgram): void {
    glc.bindBuffer(glc.ARRAY_BUFFER, buf);
    const a = attrib(p);
    glc.enableVertexAttribArray(a);
    glc.vertexAttribPointer(a, 2, glc.FLOAT, false, 0, 0);
  }

  function makeTarget(w: number, h: number): RenderTarget | null {
    const tex = glc.createTexture();
    const fb = glc.createFramebuffer();
    if (!tex || !fb) return null;
    glc.bindTexture(glc.TEXTURE_2D, tex);
    glc.texImage2D(glc.TEXTURE_2D, 0, glc.RGBA, w, h, 0, glc.RGBA, glc.UNSIGNED_BYTE, null);
    glc.texParameteri(glc.TEXTURE_2D, glc.TEXTURE_MIN_FILTER, glc.LINEAR);
    glc.texParameteri(glc.TEXTURE_2D, glc.TEXTURE_MAG_FILTER, glc.LINEAR);
    glc.texParameteri(glc.TEXTURE_2D, glc.TEXTURE_WRAP_S, glc.CLAMP_TO_EDGE);
    glc.texParameteri(glc.TEXTURE_2D, glc.TEXTURE_WRAP_T, glc.CLAMP_TO_EDGE);
    glc.bindFramebuffer(glc.FRAMEBUFFER, fb);
    glc.framebufferTexture2D(glc.FRAMEBUFFER, glc.COLOR_ATTACHMENT0, glc.TEXTURE_2D, tex, 0);
    const ok = glc.checkFramebufferStatus(glc.FRAMEBUFFER) === glc.FRAMEBUFFER_COMPLETE;
    glc.bindFramebuffer(glc.FRAMEBUFFER, null);
    return { fb, tex, w, h, ok };
  }

  let W = 0;
  let H = 0;
  let bW = 0;
  let bH = 0;
  let scene: RenderTarget | null = null;
  let bloomA: RenderTarget | null = null;
  let bloomB: RenderTarget | null = null;
  let bloomOK = Boolean(pScene && pBright && pBlur && pComp);

  function ensure(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.25);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    if (w === W && h === H && scene) return;
    W = w;
    H = h;
    bW = Math.max(1, w >> 2);
    bH = Math.max(1, h >> 2);
    for (const t of [scene, bloomA, bloomB]) {
      if (t) {
        glc.deleteTexture(t.tex);
        glc.deleteFramebuffer(t.fb);
      }
    }
    scene = makeTarget(W, H);
    bloomA = makeTarget(bW, bH);
    bloomB = makeTarget(bW, bH);
    if (!scene?.ok || !bloomA?.ok || !bloomB?.ok) bloomOK = false;
  }

  function u(p: WebGLProgram, n: string): WebGLUniformLocation | null {
    return glc.getUniformLocation(p, n);
  }

  function drawScene(target: RenderTarget | null, ts: number): void {
    glc.bindFramebuffer(glc.FRAMEBUFFER, target ? target.fb : null);
    glc.viewport(0, 0, target ? target.w : W, target ? target.h : H);
    glc.useProgram(sceneProg);
    bindQuad(sceneProg);
    glc.uniform2f(u(sceneProg, 'u_res'), target ? target.w : W, target ? target.h : H);
    glc.uniform1f(u(sceneProg, 'u_time'), ts);
    glc.uniform1f(u(sceneProg, 'u_dim'), dim);
    glc.uniform1f(u(sceneProg, 'u_scale'), scale);
    glc.drawArrays(glc.TRIANGLES, 0, 3);
  }

  function render(ts: number): void {
    ensure();
    if (!bloomOK || !pBright || !pBlur || !pComp || !scene || !bloomA || !bloomB) {
      drawScene(null, ts);
      return;
    }

    // 1) scene -> texture
    drawScene(scene, ts);

    // 2) bright pass -> bloomA (downscaled)
    glc.bindFramebuffer(glc.FRAMEBUFFER, bloomA.fb);
    glc.viewport(0, 0, bW, bH);
    glc.useProgram(pBright);
    bindQuad(pBright);
    glc.activeTexture(glc.TEXTURE0);
    glc.bindTexture(glc.TEXTURE_2D, scene.tex);
    glc.uniform1i(u(pBright, 'u_tex'), 0);
    glc.uniform1f(u(pBright, 'u_threshold'), threshold);
    glc.uniform1f(u(pBright, 'u_knee'), 0.28);
    glc.drawArrays(glc.TRIANGLES, 0, 3);

    // 3) separable gaussian blur, ping-pong A<->B
    glc.useProgram(pBlur);
    bindQuad(pBlur);
    glc.uniform2f(u(pBlur, 'u_texel'), 1.0 / bW, 1.0 / bH);
    glc.uniform1f(u(pBlur, 'u_spread'), spread);
    let src = bloomA;
    let dst = bloomB;
    for (let i = 0; i < passes; i += 1) {
      const horiz = i % 2 === 0;
      glc.bindFramebuffer(glc.FRAMEBUFFER, dst.fb);
      glc.viewport(0, 0, bW, bH);
      glc.activeTexture(glc.TEXTURE0);
      glc.bindTexture(glc.TEXTURE_2D, src.tex);
      glc.uniform1i(u(pBlur, 'u_tex'), 0);
      glc.uniform2f(u(pBlur, 'u_dir'), horiz ? 1 : 0, horiz ? 0 : 1);
      glc.drawArrays(glc.TRIANGLES, 0, 3);
      const tmp = src;
      src = dst;
      dst = tmp;
    }

    // 4) composite scene + bloom -> screen
    glc.bindFramebuffer(glc.FRAMEBUFFER, null);
    glc.viewport(0, 0, W, H);
    glc.useProgram(pComp);
    bindQuad(pComp);
    glc.activeTexture(glc.TEXTURE0);
    glc.bindTexture(glc.TEXTURE_2D, scene.tex);
    glc.uniform1i(u(pComp, 'u_scene'), 0);
    glc.activeTexture(glc.TEXTURE1);
    glc.bindTexture(glc.TEXTURE_2D, src.tex);
    glc.uniform1i(u(pComp, 'u_bloom'), 1);
    glc.uniform1f(u(pComp, 'u_intensity'), bloom);
    glc.uniform1f(u(pComp, 'u_exposure'), exposure);
    glc.drawArrays(glc.TRIANGLES, 0, 3);
  }

  let lastTime = 8.0;
  const onResize = (): void => render(lastTime);
  window.addEventListener('resize', onResize);

  const draw = (time: number): void => {
    lastTime = time;
    render(time);
  };

  let stopAnimator: () => void;
  if (animate) {
    stopAnimator = makeAnimator(canvas, speed, 8.0, draw);
  } else {
    draw(8.0);
    stopAnimator = () => undefined;
  }

  return {
    destroy: () => {
      stopAnimator();
      window.removeEventListener('resize', onResize);
      for (const t of [scene, bloomA, bloomB]) {
        if (t) {
          glc.deleteTexture(t.tex);
          glc.deleteFramebuffer(t.fb);
        }
      }
      if (buf) glc.deleteBuffer(buf);
    },
  };
}

interface Ember {
  x: number;
  y: number;
  r: number;
  vy: number;
  sway: number;
  swayA: number;
  a: number;
}

export function initEmbers(canvas: HTMLCanvasElement, opts: EmberOptions = {}): LavaController | null {
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  const c2d: CanvasRenderingContext2D = ctx;
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  const count = opts.count ?? 46;
  let W = 0;
  let H = 0;
  let parts: Ember[] = [];

  function resize(): void {
    W = canvas.clientWidth;
    H = canvas.clientHeight;
    canvas.width = Math.floor(W * dpr);
    canvas.height = Math.floor(H * dpr);
    c2d.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function spawn(y?: number): Ember {
    return {
      x: Math.random() * W,
      y: y ?? Math.random() * H,
      r: 0.6 + Math.random() * 1.8,
      vy: 0.15 + Math.random() * 0.5,
      sway: Math.random() * Math.PI * 2,
      swayA: 0.2 + Math.random() * 0.5,
      a: 0.15 + Math.random() * 0.5,
    };
  }

  resize();
  parts = Array.from({ length: count }, () => spawn());

  function frame(): void {
    c2d.clearRect(0, 0, W, H);
    c2d.globalCompositeOperation = 'lighter';
    for (let i = 0; i < parts.length; i += 1) {
      const p = parts[i];
      p.y -= p.vy;
      p.sway += 0.01;
      const x = p.x + Math.sin(p.sway) * p.swayA;
      if (p.y < -4) {
        parts[i] = spawn(H + 4);
        continue;
      }
      const g = c2d.createRadialGradient(x, p.y, 0, x, p.y, p.r * 3);
      g.addColorStop(0, `rgba(255,238,200,${p.a})`);
      g.addColorStop(0.4, `rgba(255,150,60,${p.a * 0.5})`);
      g.addColorStop(1, 'rgba(255,120,40,0)');
      c2d.fillStyle = g;
      c2d.beginPath();
      c2d.arc(x, p.y, p.r * 3, 0, Math.PI * 2);
      c2d.fill();
    }
    c2d.globalCompositeOperation = 'source-over';
  }

  const onResize = (): void => {
    resize();
    frame();
  };
  window.addEventListener('resize', onResize);

  const stopAnimator = makeAnimator(canvas, 1, 0, () => frame());

  return {
    destroy: () => {
      stopAnimator();
      window.removeEventListener('resize', onResize);
    },
  };
}
