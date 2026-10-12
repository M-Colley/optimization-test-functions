import test from 'node:test';
import assert from 'node:assert/strict';
import {
  functions, optimizationBenchmarks, psychophysicalLaws,
  domain, fromUnitCube, negate, branin, ackley, shekel, michalewicz, hickHyman, stevens,
} from '../src/index.js';

/* A dimension each function actually accepts. */
const DIM = {
  ackley: 4, griewank: 4, schwefel: 4, eggholder: 2, powell: 4, shekel: 4,
  hartmann3: 3, hartmann6: 6, branin: 2, rosenbrock: 4, rastrigin: 4,
  michalewicz: 5, yerkesDodson: 3, stevens: 3, hickHyman: 3, weberFechner: 3,
};

const names = Object.keys(functions);

/* Stated optima must hold to a few ulps, not to the five or six digits the
   literature usually quotes: a looser check is how a value below the true
   minimum goes unnoticed. */
const close = (got, want) => Math.abs(got - want) <= 1e-12 * Math.max(1, Math.abs(want));

/* mulberry32 (Ettinger, public domain): exact 32-bit integer arithmetic via
   Math.imul. A float LCG such as (seed * 1103515245 + 12345) & 0x7FFFFFFF
   loses its low bits to double rounding and collapses onto ~10^4 values. */
function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* Compass search from x0, kept inside the box: the best value it reaches. */
function polish(f, x0, box, sense) {
  const s = sense === 'min' ? 1 : -1;
  const x = Array.from(x0);
  let fx = s * f(x);
  for (let h = 1e-3; h > 1e-13; h /= 2) {
    for (let improved = true; improved;) {
      improved = false;
      for (let j = 0; j < x.length; j++) {
        for (const dir of [1, -1]) {
          const old = x[j], w = box[j][1] - box[j][0];
          x[j] = Math.min(box[j][1], Math.max(box[j][0], old + dir * h * w));
          const v = s * f(x);
          if (v < fx) { fx = v; improved = true; } else x[j] = old;
        }
      }
    }
  }
  return s * fx;
}

test('the registry and the two groupings agree', () => {
  assert.equal(names.length, 16);
  assert.deepEqual(
    [...optimizationBenchmarks, ...psychophysicalLaws].sort(),
    names.slice().sort(),
  );
});

test('every function carries usable metadata', () => {
  for (const n of names) {
    const m = functions[n].meta;
    assert.ok(m, `${n} has meta`);
    assert.ok(m.name && m.reference, `${n} names itself and its source`);
    assert.ok(m.sense === 'min' || m.sense === 'max', `${n} declares a sense`);
    const box = domain(n, DIM[n]);
    assert.equal(box.length, DIM[n], `${n} domain has one range per axis`);
    for (const [lo, hi] of box) assert.ok(hi > lo, `${n} range is non-empty`);
  }
});

test('the stated optimum is attained, to the ulp, at the stated location', () => {
  for (const n of names) {
    const f = functions[n], m = f.meta;
    if (m.optimum.value === null) continue;          // michalewicz: d-dependent
    const box = domain(n, DIM[n]);
    for (const at of [m.optimum.at(DIM[n]), ...(m.optimum.all ?? [])]) {
      at.forEach((v, j) => assert.ok(v >= box[j][0] && v <= box[j][1], `${n}: optimum outside its domain`));
      const got = f(at);
      assert.ok(close(got, m.optimum.value),
        `${n}: f(${at}) = ${got}, meta says ${m.optimum.value}`);
    }
  }
});

test('the stated optimum cannot be improved by local search from it', () => {
  // Catches a value that is attained but is not the minimum, e.g. f at a
  // rounded location rather than at the optimizer itself.
  for (const n of names) {
    const f = functions[n], m = f.meta;
    if (m.optimum.value === null) continue;
    const box = domain(n, DIM[n]), want = m.optimum.value;
    const got = polish(f, m.optimum.at(DIM[n]), box, m.sense);
    const better = m.sense === 'min' ? got < want : got > want;
    assert.ok(!better || close(got, want), `${n}: local search reached ${got}, past the stated ${want}`);
  }
});

test('the stated optimum is not beaten by 200k random points in the domain', () => {
  const rnd = mulberry32(20260826);
  for (const n of names) {
    const f = functions[n], m = f.meta;
    if (m.optimum.value === null) continue;
    const d = DIM[n], box = domain(n, d), best = m.optimum.value;
    const better = m.sense === 'min' ? (v) => v < best - 1e-9 : (v) => v > best + 1e-9;
    const x = new Array(d);
    for (let k = 0; k < 200000; k++) {
      for (let j = 0; j < d; j++) x[j] = box[j][0] + rnd() * (box[j][1] - box[j][0]);
      const v = f(x);
      assert.ok(Number.isFinite(v), `${n} returned ${v}`);
      assert.ok(!better(v), `${n}: random search found ${v}, beating the stated ${best}`);
    }
  }
});

test("Branin's three global minima are all equal, on its asymmetric box", () => {
  const box = domain('branin');
  assert.deepEqual(box, [[-5, 10], [0, 15]]);
  assert.equal(branin.meta.optimum.all.length, 3);
  for (const at of branin.meta.optimum.all) {
    assert.ok(close(branin(at), 5 / (4 * Math.PI)), `branin${JSON.stringify(at)} = ${branin(at)}`);
  }
});

test("michalewicz's known optima are exact, by separability", () => {
  // f(x) = -sum_i g_i(x_i), so the d-D minimum is the sum of the 1-D minima.
  // Bracket each g_i on a dense grid, refine by golden section, and check
  // f at the assembled argmin against every tabulated d.
  const { known } = michalewicz.meta.optimum;
  assert.deepEqual(Object.keys(known).map(Number), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  const r = (Math.sqrt(5) - 1) / 2, n = 100000, h = Math.PI / n, at = [];
  for (let i = 1; i <= 10; i++) {
    const g = (t) => Math.sin(t) * Math.sin(i * t * t / Math.PI) ** 20;
    let k0 = 0;
    for (let k = 1; k <= n; k++) if (g(k * h) > g(k0 * h)) k0 = k;
    let lo = Math.max(0, (k0 - 1) * h), hi = Math.min(Math.PI, (k0 + 1) * h);
    let a = hi - r * (hi - lo), b = lo + r * (hi - lo), ga = g(a), gb = g(b);
    for (let it = 0; it < 80; it++) {
      if (ga > gb) { hi = b; b = a; gb = ga; a = hi - r * (hi - lo); ga = g(a); }
      else { lo = a; a = b; ga = gb; b = lo + r * (hi - lo); gb = g(b); }
    }
    at.push((lo + hi) / 2);
    assert.ok(close(michalewicz(at), known[i]), `d=${i}: f(argmin) = ${michalewicz(at)}, meta says ${known[i]}`);
  }
  // and they round to the values the literature quotes
  assert.equal(known[2].toFixed(4), '-1.8013');
  assert.equal(known[5].toFixed(6), '-4.687658');
  assert.equal(known[10].toFixed(5), '-9.66015');
});

test('dimension validation rejects wrong input', () => {
  assert.throws(() => branin([1, 2, 3]), RangeError);
  assert.throws(() => shekel([1, 2, 3]), RangeError);
  assert.throws(() => functions.powell([1, 2, 3]), RangeError);   // not a multiple of 4
  assert.throws(() => ackley('nope'), TypeError);
  assert.throws(() => domain('branin', 3), RangeError);            // fixed 2-D
  assert.throws(() => domain('ackley', 0), RangeError);
  assert.throws(() => domain('ackley', 2.5), RangeError);
  assert.throws(() => fromUnitCube('branin', [0.5]), RangeError);
});

test('negate flips the value and all of the metadata that depends on its sign', () => {
  const x = [0.3, 0.4, 0.5, 0.6];
  const g = negate(ackley);
  assert.equal(g(x), -ackley(x));
  assert.equal(ackley.meta.sense, 'min');
  assert.equal(g.meta.sense, 'max');

  const nb = negate(branin);
  assert.equal(nb.meta.optimum.value, -branin.meta.optimum.value);
  assert.ok(close(nb(nb.meta.optimum.at()), nb.meta.optimum.value));

  const ns = negate(stevens);
  assert.equal(ns.meta.sense, 'min');
  assert.equal(ns.meta.monotone, 'decreasing');
  assert.equal(stevens.meta.monotone, 'increasing');               // original untouched

  const nm = negate(michalewicz);
  assert.equal(nm.meta.optimum.value, null);
  assert.equal(nm.meta.optimum.known[2], -michalewicz.meta.optimum.known[2]);

  assert.deepEqual(negate(negate(branin)).meta, branin.meta);
});

test('fromUnitCube respects a per-axis domain', () => {
  assert.deepEqual(fromUnitCube('branin', [0, 0]), [-5, 0]);
  assert.deepEqual(fromUnitCube('branin', [1, 1]), [10, 15]);
  const mid = fromUnitCube('branin', [0.5, 0.5]);
  assert.deepEqual(mid, [2.5, 7.5]);
});

test('fromUnitCube returns plain doubles for typed-array input', () => {
  const u = new Float32Array([0.1, 0.7]);
  const p = fromUnitCube('branin', u);
  assert.ok(Array.isArray(p));
  assert.deepEqual(p, [-5 + u[0] * 15, u[1] * 15]);
});

test('hickHyman returns a time, so lower is better', () => {
  assert.equal(hickHyman.meta.sense, 'min');
  assert.ok(hickHyman([0, 0, 0]) < hickHyman([10, 10, 10]));
});

test('optional parameters are honoured', () => {
  assert.notEqual(shekel([4, 4, 4, 4], { m: 5 }), shekel([4, 4, 4, 4], { m: 10 }));
  assert.notEqual(michalewicz([1, 2], { m: 1 }), michalewicz([1, 2], { m: 10 }));
  assert.notEqual(ackley([1, 1], { a: 25 }), ackley([1, 1]));
  assert.throws(() => shekel([4, 4, 4, 4], { m: 5.5 }), RangeError);
  assert.throws(() => shekel([4, 4, 4, 4], { m: 11 }), RangeError);
});
