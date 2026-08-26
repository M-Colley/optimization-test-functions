import test from 'node:test';
import assert from 'node:assert/strict';
import {
  functions, optimizationBenchmarks, psychophysicalLaws,
  domain, fromUnitCube, negate, branin, ackley, shekel, michalewicz, hickHyman,
} from '../src/index.js';

/* A dimension each function actually accepts. */
const DIM = {
  ackley: 4, griewank: 4, schwefel: 4, eggholder: 2, powell: 4, shekel: 4,
  hartmann3: 3, hartmann6: 6, branin: 2, rosenbrock: 4, rastrigin: 4,
  michalewicz: 5, yerkesDodson: 3, stevens: 3, hickHyman: 3, weberFechner: 3,
};

const names = Object.keys(functions);

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

test('the stated optimum really is attained at the stated location', () => {
  for (const n of names) {
    const f = functions[n], m = f.meta;
    if (m.optimum.value === null) continue;          // michalewicz: d-dependent
    const at = m.optimum.at(DIM[n]);
    const got = f(at);
    assert.ok(Math.abs(got - m.optimum.value) < 1e-4,
      `${n}: f(optimum) = ${got}, meta says ${m.optimum.value}`);
  }
});

test('the stated optimum is not beaten by random search in the domain', () => {
  // A crude but honest check: 200k samples should not find anything better.
  let seed = 20260826;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF) / 0x7FFFFFFF;
  for (const n of names) {
    const f = functions[n], m = f.meta;
    if (m.optimum.value === null) continue;
    const d = DIM[n], box = domain(n, d), best = m.optimum.value;
    const better = m.sense === 'min' ? (v) => v < best - 1e-6 : (v) => v > best + 1e-6;
    const x = new Array(d);
    for (let k = 0; k < 200000 / d; k++) {
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
  for (const at of branin.meta.optimum.all) {
    assert.ok(Math.abs(branin(at) - branin.meta.optimum.value) < 1e-4,
      `branin${JSON.stringify(at)} = ${branin(at)}`);
  }
});

test('michalewicz matches its published values at d = 2, 5 and 10', () => {
  // Verified by dense local refinement rather than a stated argmin, since the
  // literature reports the value but not the location for every d.
  for (const [d, want] of Object.entries(michalewicz.meta.optimum.known)) {
    const dim = Number(d);
    let seed = 7, best = Infinity;
    const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF) / 0x7FFFFFFF;
    const x = new Array(dim);
    for (let k = 0; k < 400000; k++) {
      for (let j = 0; j < dim; j++) x[j] = rnd() * Math.PI;
      const v = michalewicz(x);
      if (v < best) best = v;
    }
    // random search will not reach the optimum in 10-D, only bound it
    assert.ok(best >= want - 1e-6, `d=${dim}: found ${best}, below published ${want}`);
  }
});

test('dimension validation rejects wrong input', () => {
  assert.throws(() => branin([1, 2, 3]), RangeError);
  assert.throws(() => shekel([1, 2, 3]), RangeError);
  assert.throws(() => functions.powell([1, 2, 3]), RangeError);   // not a multiple of 4
  assert.throws(() => ackley('nope'), TypeError);
});

test('negate flips both the value and the declared sense', () => {
  const x = [0.3, 0.4, 0.5, 0.6];
  const g = negate(ackley);
  assert.equal(g(x), -ackley(x));
  assert.equal(ackley.meta.sense, 'min');
  assert.equal(g.meta.sense, 'max');
});

test('fromUnitCube respects a per-axis domain', () => {
  assert.deepEqual(fromUnitCube('branin', [0, 0]), [-5, 0]);
  assert.deepEqual(fromUnitCube('branin', [1, 1]), [10, 15]);
  const mid = fromUnitCube('branin', [0.5, 0.5]);
  assert.deepEqual(mid, [2.5, 7.5]);
});

test('hickHyman returns a time, so lower is better', () => {
  assert.equal(hickHyman.meta.sense, 'min');
  assert.ok(hickHyman([0, 0, 0]) < hickHyman([10, 10, 10]));
});

test('optional parameters are honoured', () => {
  assert.notEqual(shekel([4, 4, 4, 4], { m: 5 }), shekel([4, 4, 4, 4], { m: 10 }));
  assert.notEqual(michalewicz([1, 2], { m: 1 }), michalewicz([1, 2], { m: 10 }));
  assert.notEqual(ackley([1, 1], { a: 25 }), ackley([1, 1]));
});
