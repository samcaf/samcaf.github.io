/**
 * galaxy.js — the concept graph, rendered as a slowly rotating galaxy.
 *
 * The layout is precomputed in 3D by tools/build_graph.mjs, so the browser only
 * projects and spins it: no physics at runtime, no dependency, ~150 lines of
 * canvas 2D. What lights up is exactly the set of concepts retrieval touched —
 * the picture is a readout of the search, not decoration.
 */

import { prepareGraph } from './graphboost.js';

const GRAPH_URL = new URL('../../rag/graph.json', import.meta.url).href;

let graphPromise = null;
export function loadGraph() {
  if (!graphPromise) {
    graphPromise = (async () => {
      const r = await fetch(GRAPH_URL);
      if (!r.ok) throw new Error(`graph unavailable (${r.status})`);
      const g = await r.json();
      // Concept embeddings live in a sidecar, same as the corpus vectors.
      if (g.vectors?.file) {
        const v = await fetch(new URL(g.vectors.file, GRAPH_URL).href);
        if (v.ok) g.vecs = new Int8Array(await v.arrayBuffer());
      }
      return prepareGraph(g);   // one owner of graph indexing, in graphboost.js
    })();
  }
  return graphPromise;
}

const TAU = Math.PI * 2;
const TILT = 0.42;      // fixed 3/4 view so the disc reads as a disc
const DEPTH = 2.8;      // perspective distance

export function createGalaxy(canvas, { onSelect } = {}) {
  const ctx = canvas.getContext('2d');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  let graph = null;
  let proj = [];             // per-node screen position + depth
  let active = new Set();
  let halo = new Set();      // one hop out from active
  let hover = -1;
  let theta = 0.6;
  let targetTheta = null;
  let raf = 0;
  let w = 0;
  let h = 0;
  let dpr = 1;
  let drag = null;

  const css = (name, fallback) =>
    getComputedStyle(canvas).getPropertyValue(name).trim() || fallback;

  function resize() {
    const rect = canvas.getBoundingClientRect();
    dpr = Math.min(devicePixelRatio || 1, 2);
    w = Math.max(1, rect.width);
    h = Math.max(1, rect.height);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function project() {
    const cx = w / 2;
    const cy = h / 2;
    const R = Math.min(w, h) * 0.44;
    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    const ct = Math.cos(TILT);
    const st = Math.sin(TILT);

    proj = graph.nodes.map((n) => {
      const [x, y, z] = n.p;
      const rx = x * cos + z * sin;
      const rz = -x * sin + z * cos;
      const ry = y * ct - rz * st;
      const rz2 = y * st + rz * ct;
      const k = DEPTH / (DEPTH + rz2);
      return { x: cx + rx * R * k, y: cy + ry * R * k, k };
    });
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);
    if (!graph) return;
    project();

    const cEdge = css('--border-strong', 'rgba(255,255,255,0.16)');
    const cPub = css('--accent', '#8b9dff');
    const cNote = css('--accent-3', '#c08bff');
    const cHot = css('--accent-2', '#56d9e0');
    const cText = css('--text', '#eef0f6');
    const dim = active.size ? 0.28 : 1;

    // Edges first, so stars sit on top of their own filaments.
    ctx.lineWidth = 1;
    for (const e of graph.edges) {
      const a = proj[e.a];
      const b = proj[e.b];
      const lit = (active.has(e.a) && (active.has(e.b) || halo.has(e.b)))
        || (active.has(e.b) && halo.has(e.a));
      const depth = (a.k + b.k) / 2;
      ctx.globalAlpha = (lit ? 0.55 : 0.13 * dim) * Math.min(1, depth * 0.75);
      ctx.strokeStyle = lit ? cHot : cEdge;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }

    // Stars, far to near.
    const order = graph.nodes.map((_, i) => i).sort((i, j) => proj[i].k - proj[j].k);
    for (const i of order) {
      const n = graph.nodes[i];
      const p = proj[i];
      const isActive = active.has(i);
      const isHalo = halo.has(i);
      const base = 1.1 + Math.min(2.4, Math.log2(1 + n.df) * 0.42);
      const r = base * p.k * (isActive ? 1.9 : 1);

      ctx.globalAlpha = (isActive ? 1 : isHalo ? 0.7 : 0.42 * dim) * Math.min(1, p.k * 0.8);
      ctx.fillStyle = isActive ? cHot : n.corpus === 'notes' ? cNote : cPub;
      ctx.shadowBlur = isActive ? 14 * p.k : 0;
      ctx.shadowColor = isActive ? cHot : 'transparent';
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, TAU);
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // Labels only where they mean something: lit concepts and whatever is hovered.
    ctx.font = `500 11px ${css('--font-body', 'system-ui')}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    const labelled = [...active].slice(0, 6);
    if (hover >= 0 && !labelled.includes(hover)) labelled.push(hover);
    for (const i of labelled) {
      const p = proj[i];
      if (p.k < 0.55) continue;
      const text = graph.nodes[i].name;
      ctx.globalAlpha = 0.92;
      ctx.fillStyle = css('--bg', '#0a0b10');
      const wTxt = ctx.measureText(text).width;
      ctx.fillRect(p.x - wTxt / 2 - 4, p.y - 20, wTxt + 8, 15);
      ctx.fillStyle = i === hover ? cHot : cText;
      ctx.fillText(text, p.x, p.y - 8);
    }
    ctx.globalAlpha = 1;
  }

  function frame() {
    if (targetTheta !== null) {
      let d = ((targetTheta - theta + Math.PI) % TAU + TAU) % TAU - Math.PI;
      if (Math.abs(d) < 0.005) targetTheta = null;
      else theta += d * 0.06;
    } else if (!reduced && !drag) {
      theta += 0.0022;
    }
    draw();
    raf = requestAnimationFrame(frame);
  }

  function pick(ev) {
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    let best = -1;
    let bestD = 15;
    proj.forEach((p, i) => {
      const d = Math.hypot(p.x - mx, p.y - my);
      if (d < bestD) { bestD = d; best = i; }
    });
    return best;
  }

  const onMove = (ev) => {
    if (drag) {
      theta -= (ev.clientX - drag.x) * 0.006;
      drag.x = ev.clientX;
      targetTheta = null;
      return;
    }
    const was = hover;
    hover = pick(ev);
    canvas.style.cursor = hover >= 0 ? 'pointer' : 'grab';
    if (hover !== was && graph) canvas.title = hover >= 0 ? graph.nodes[hover].name : '';
  };
  const onDown = (ev) => { drag = { x: ev.clientX, moved: 0 }; canvas.setPointerCapture?.(ev.pointerId); };
  const onUp = (ev) => {
    const wasDrag = drag && Math.abs(ev.clientX - drag.x) > 3;
    drag = null;
    if (wasDrag || !graph) return;
    const i = pick(ev);
    if (i >= 0 && onSelect) onSelect(graph.nodes[i], i);
  };
  const onLeave = () => { hover = -1; drag = null; };

  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointerleave', onLeave);

  const ro = new ResizeObserver(() => { resize(); if (graph) draw(); });
  ro.observe(canvas);

  // Pause while off-screen — a hidden canvas has no business burning frames.
  const io = new IntersectionObserver(([entry]) => {
    if (entry.isIntersecting) { if (!raf) raf = requestAnimationFrame(frame); }
    else if (raf) { cancelAnimationFrame(raf); raf = 0; }
  }, { threshold: 0.01 });
  io.observe(canvas);

  loadGraph().then((g) => {
    graph = g;
    resize();
    draw();   // one frame now: requestAnimationFrame is starved in a hidden or
              // backgrounded tab, and a galaxy that only paints inside the loop
              // would sit blank there until the tab is focused
    if (!raf) raf = requestAnimationFrame(frame);
  });

  return {
    get graph() { return graph; },
    /** Names of the currently lit concepts — handy for debugging and tests. */
    get activeNames() { return graph ? [...active].map((i) => graph.nodes[i].name) : []; },
    /** Stop/start the animation loop (closed panel, screenshots, tests). */
    pause() { if (raf) { cancelAnimationFrame(raf); raf = 0; } draw(); },
    resume() { if (!raf) raf = requestAnimationFrame(frame); },
    /** Light the given node indices, plus a dimmer one-hop halo. */
    setActive(indices) {
      active = new Set(indices);
      halo = new Set();
      for (const i of active) for (const j of graph?.neighbors[i] ?? []) if (!active.has(j)) halo.add(j);
      // Turn the galaxy so the lit region faces the viewer.
      if (graph && active.size) {
        let sx = 0;
        let sz = 0;
        for (const i of active) { sx += graph.nodes[i].p[0]; sz += graph.nodes[i].p[2]; }
        if (sx || sz) targetTheta = -Math.atan2(sx / active.size, sz / active.size);
      }
    },
    clear() { active = new Set(); halo = new Set(); targetTheta = null; },
    destroy() {
      if (raf) cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointerup', onUp);
      canvas.removeEventListener('pointerleave', onLeave);
    },
  };
}
