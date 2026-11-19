/**
 * optimization-test-functions
 *
 * Canonical implementations of standard global-optimization benchmarks and
 * classical psychophysical laws, with the metadata you need to use them
 * honestly: the domain each is defined on, its known optimum, and whether the
 * interesting direction is a minimum or a maximum.
 *
 * Every function is in its PUBLISHED sense. The optimization benchmarks are
 * minimized, as their sources define them; the psychophysical laws are given
 * as the law states them. Nothing here is silently negated — if your optimizer
 * maximizes, wrap with `negate`.
 *
 * Domains are per-axis, not one range applied to every dimension. Branin in
 * particular is defined on an asymmetric box and is often gotten wrong.
 */

const TAU = 2 * Math.PI;

/* ------------------------------------------------------------------ *
 * helpers                                                            *
 * ------------------------------------------------------------------ */

function isVec(x) {
  return Array.isArray(x) || (
    ArrayBuffer.isView(x) &&
    !(x instanceof DataView) &&
    !(x instanceof BigInt64Array) &&
    !(x instanceof BigUint64Array)
  );
}

function assertVec(x, name) {
  if (!isVec(x)) throw new TypeError(`${name}: expected an array of numbers`);
  if (x.length < 1) throw new RangeError(`${name}: unsupported dimension ${x.length}`);
}

function assertDim(x, name, d) {
  if (!isVec(x)) throw new TypeError(`${name}: expected an array of numbers`);
  if (x.length !== d) throw new RangeError(`${name}: unsupported dimension ${x.length}`);
}

function assertOpts(options, name) {
  if (options === null || typeof options !== 'object') {
    throw new TypeError(`${name}: options must be an object`);
  }
}

function checkDimFor(name, dims, d) {
  if (typeof dims === 'number') {
    if (d !== undefined && d !== dims) throw new RangeError(`${name} is ${dims}-D, not ${d}-D`);
    return dims;
  }
  if (d === undefined) throw new RangeError(`${name}: dimension is required`);
  if (!Number.isInteger(d) || d < 1) {
    throw new RangeError(`${name}: dimension must be a positive integer, got ${d}`);
  }
  if (dims === '>= 2' && d < 2) {
    throw new RangeError(`${name}: dimension must be >= 2, got ${d}`);
  }
  if (dims === 'multiple of 4' && (d & 3) !== 0) {
    throw new RangeError(`${name}: dimension must be a multiple of 4, got ${d}`);
  }
  return d;
}

function fillAt(name, dims, val) {
  return (d) => new Array(checkDimFor(name, dims, d)).fill(val);
}

function fixedAt(name, pt) {
  const n = pt.length;
  return (d = n) => {
    if (d !== n) throw new RangeError(`${name} is ${n}-D, not ${d}-D`);
    return pt.slice();
  };
}

/**
 * Wrap a function so a minimizer can be used as a maximizer, or vice versa.
 * The metadata follows: the sense, the optimum value (and any per-dimension
 * table of them) and the direction of monotonicity all flip with the sign.
 */
export function negate(f) {
  if (typeof f !== 'function') throw new TypeError('negate: expected a function');
  const g = (...args) => 0 - f(...args);
  if (f.meta) {
    const { sense, monotone, optimum } = f.meta;
    g.meta = {
      ...f.meta,
      sense: sense === 'min' ? 'max' : 'min',
      optimum: { ...optimum, value: optimum.value === null ? null : 0 - optimum.value },
    };
    if (optimum.known) {
      g.meta.optimum.known = Object.fromEntries(
        Object.entries(optimum.known).map(([d, v]) => [d, 0 - v]));
    }
    if (monotone) g.meta.monotone = monotone === 'increasing' ? 'decreasing' : 'increasing';
  }
  return g;
}

/** Expand a per-axis domain to `d` dimensions, for functions defined on a cube. */
function cube(lo, hi, dims = 'any', name = 'domain') {
  if (typeof dims === 'number') {
    return (d = dims) => {
      if (d !== dims) throw new RangeError(`${name} is ${dims}-D, not ${d}-D`);
      const out = new Array(dims);
      for (let i = 0; i < dims; i++) out[i] = [lo, hi];
      return out;
    };
  }
  return (d) => {
    const n = checkDimFor(name, dims, d);
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = [lo, hi];
    return out;
  };
}

/* ------------------------------------------------------------------ *
 * classical optimization benchmarks (minimized)                      *
 * ------------------------------------------------------------------ */

/** Ackley (1987). Flat outer plain, narrow basin at the origin. */
export function ackley(x, options) {
  assertVec(x, 'ackley');
  let a = 20, b = 0.2, c = TAU;
  if (options !== undefined) {
    assertOpts(options, 'ackley');
    if (options.a !== undefined) a = options.a;
    if (options.b !== undefined) b = options.b;
    if (options.c !== undefined) c = options.c;
  }
  const d = x.length;
  let s1 = 0, s2 = 0;
  for (let i = 0; i < d; i++) {
    const xi = x[i];
    s1 += xi * xi;
    s2 += Math.cos(c * xi);
  }
  return (a - a * Math.exp(-b * Math.sqrt(s1 / d))) + (Math.E - Math.exp(s2 / d));
}
ackley.meta = {
  name: 'Ackley', dims: 'any', sense: 'min', domain: cube(-32.768, 32.768, 'any', 'ackley'),
  optimum: { value: 0, at: fillAt('ackley', 'any', 0) },
  reference: 'Ackley, D. H. (1987). A Connectionist Machine for Genetic Hillclimbing.',
};

const GRIEWANK_SQRT = Float64Array.from({ length: 64 }, (_, i) => Math.sqrt(i + 1));

/** Griewank (1981). Wide bowl with fine cosine ripples. */
export function griewank(x) {
  assertVec(x, 'griewank');
  const d = x.length;
  let s = 0, p = 1;
  const fast = d <= 64 ? d : 64;
  let i = 0;
  for (; i < fast; i++) {
    const xi = x[i];
    s += xi * xi;
    p *= Math.cos(xi / GRIEWANK_SQRT[i]);
  }
  for (; i < d; i++) {
    const xi = x[i];
    s += xi * xi;
    p *= Math.cos(xi / Math.sqrt(i + 1));
  }
  return s / 4000 + (1 - p);
}
griewank.meta = {
  name: 'Griewank', dims: 'any', sense: 'min', domain: cube(-600, 600, 'any', 'griewank'),
  optimum: { value: 0, at: fillAt('griewank', 'any', 0) },
  reference: 'Griewank, A. O. (1981). Journal of Optimization Theory and Applications 34(1).',
};

/** Schwefel (1981). Global optimum far from the deepest local one. */
export function schwefel(x) {
  assertVec(x, 'schwefel');
  const d = x.length;
  let s = 0;
  // max of t sin(sqrt t): t = u^2 where tan(u) = -u/2, solved at 50 digits.
  // Subtract per coordinate so large-d summation does not drift off 0.
  for (let i = 0; i < d; i++) {
    const xi = x[i];
    s += 418.9828872724337 - xi * Math.sin(Math.sqrt(Math.abs(xi)));
  }
  return s;
}
schwefel.meta = {
  name: 'Schwefel', dims: 'any', sense: 'min', domain: cube(-500, 500, 'any', 'schwefel'),
  optimum: { value: 0, at: fillAt('schwefel', 'any', 420.96874635998205) },
  reference: 'Schwefel, H.-P. (1981). Numerical Optimization of Computer Models.',
};

/** Eggholder. Two dimensions, wildly multimodal, no usable global structure. */
export function eggholder(x) {
  assertDim(x, 'eggholder', 2);
  const a = x[0], b = x[1], b47 = b + 47;
  return -b47 * Math.sin(Math.sqrt(Math.abs(a / 2 + b47)))
         - a * Math.sin(Math.sqrt(Math.abs(a - b47)));
}
eggholder.meta = {
  name: 'Eggholder', dims: 2, sense: 'min', domain: cube(-512, 512, 2, 'eggholder'),
  optimum: { value: -959.6406627208509, at: fixedAt('eggholder', [512, 404.2318051137578]) },
  reference: 'Vanaret et al. (2014); after Whitley et al. (1996).',
};

/** Powell (1962). Smooth quartic valley. Dimension must be a multiple of 4. */
export function powell(x) {
  if (!isVec(x)) throw new TypeError('powell: expected an array of numbers');
  const d = x.length;
  if (d === 0 || (d & 3) !== 0) throw new RangeError(`powell: unsupported dimension ${d}`);
  let s = 0;
  for (let i = 0; i < d; i += 4) {
    const x0 = x[i], x1 = x[i + 1], x2 = x[i + 2], x3 = x[i + 3];
    const t1 = x0 + 10 * x1;
    const t2 = x2 - x3;
    const t3 = x1 - 2 * x2, t3_2 = t3 * t3;
    const t4 = x0 - x3, t4_2 = t4 * t4;
    s += t1 * t1 + 5 * (t2 * t2) + t3_2 * t3_2 + 10 * (t4_2 * t4_2);
  }
  return s;
}
powell.meta = {
  name: 'Powell', dims: 'multiple of 4', sense: 'min', domain: cube(-4, 5, 'multiple of 4', 'powell'),
  optimum: { value: 0, at: fillAt('powell', 'multiple of 4', 0) },
  reference: 'Powell, M. J. D. (1962). The Computer Journal 5(2).',
};

const SHEKEL_A = new Float64Array([
  4, 4, 4, 4,  1, 1, 1, 1,  8, 8, 8, 8,  6, 6, 6, 6,  3, 7, 3, 7,
  2, 9, 2, 9,  5, 3, 5, 3,  8, 1, 8, 1,  6, 2, 6, 2,  7, 3.6, 7, 3.6,
]);
const SHEKEL_C = new Float64Array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]);

/** Shekel. Four dimensions, `m` narrow wells on a flat plain. */
export function shekel(x, options) {
  assertDim(x, 'shekel', 4);
  let m = 10;
  if (options !== undefined) {
    assertOpts(options, 'shekel');
    if (options.m !== undefined) m = options.m;
    if (!(Number.isInteger(m) && m >= 1 && m <= 10)) {
      throw new RangeError('shekel: m must be an integer in 1..10');
    }
  }
  const x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];
  let s = 0;
  for (let i = 0, k = 0; i < m; i++, k += 4) {
    const d0 = x0 - SHEKEL_A[k];
    const d1 = x1 - SHEKEL_A[k + 1];
    const d2 = x2 - SHEKEL_A[k + 2];
    const d3 = x3 - SHEKEL_A[k + 3];
    s += 1 / (d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 + SHEKEL_C[i]);
  }
  return -s;
}
shekel.meta = {
  name: 'Shekel', dims: 4, sense: 'min', domain: cube(0, 10, 4, 'shekel'),
  optimum: {
    value: -10.536443153483528,
    // Almost always quoted as (4, 4, 4, 4), which is only approximate: the
    // neighbouring wells pull the minimum slightly off that corner, and
    // f(4,4,4,4) = -10.536284, not -10.536443. Refined location below; it
    // has x1 = x3 and x2 = x4, as the symmetry of the wells requires.
    at: fixedAt('shekel', [4.000746868270634, 3.9995094800857736, 4.000746868270634, 3.9995094800857736]),
    approx: [4, 4, 4, 4],
    note: 'for m = 10',
  },
  reference: 'Shekel, J. (1971). Test functions for multimodal search techniques. '
           + 'Fifth Annual Princeton Conf. on Information Science and Systems.',
};

const H_ALPHA = new Float64Array([1.0, 1.2, 3.0, 3.2]);
const H3_A = new Float64Array([3.0, 10, 30, 0.1, 10, 35, 3.0, 10, 30, 0.1, 10, 35]);
// v / 1e4 rather than v * 1e-4: 1e-4 is inexact, and the product is 1 ulp off
// the decimal constant for a third of the entries; the quotient never is.
const H3_P = Float64Array.from(
  [3689, 1170, 2673, 4699, 4387, 7470, 1091, 8732, 5547, 381, 5743, 8828],
  (v) => v / 1e4,
);

/** Hartmann 3-D. Four Gaussian basins of unequal depth. */
export function hartmann3(x) {
  assertDim(x, 'hartmann3', 3);
  const x0 = x[0], x1 = x[1], x2 = x[2];
  let s = 0;
  for (let i = 0, k = 0; i < 4; i++, k += 3) {
    const d0 = x0 - H3_P[k], d1 = x1 - H3_P[k + 1], d2 = x2 - H3_P[k + 2];
    s += H_ALPHA[i] * Math.exp(-(H3_A[k] * d0 * d0 + H3_A[k + 1] * d1 * d1 + H3_A[k + 2] * d2 * d2));
  }
  return -s;
}
hartmann3.meta = {
  name: 'Hartmann-3', dims: 3, sense: 'min', domain: cube(0, 1, 3, 'hartmann3'),
  optimum: {
    value: -3.8627797873326624,
    // The widely copied -3.86278214782076 is 2.4e-6 BELOW the true minimum,
    // so no point attains it, and the usual location (kept as `approx`) is
    // 2.5e-5 off in x1. Both re-derived at 50 digits.
    at: fixedAt('hartmann3', [0.11458887665506896, 0.55564889461693, 0.8525469846866774]),
    approx: [0.114614, 0.555649, 0.852547],
  },
  reference: 'Hartman, J. K. (1973). Naval Research Logistics Quarterly 20(3).',
};

const H6_A = new Float64Array([
  10, 3, 17, 3.5, 1.7, 8,  0.05, 10, 17, 0.1, 8, 14,
  3, 3.5, 1.7, 10, 17, 8,  17, 8, 0.05, 10, 0.1, 14,
]);
const H6_P = Float64Array.from([
  1312, 1696, 5569, 124, 8283, 5886,  2329, 4135, 8307, 3736, 1004, 9991,
  2348, 1451, 3522, 2883, 3047, 6650,  4047, 8828, 8732, 5743, 1091, 381,
], (v) => v / 1e4);

/** Hartmann 6-D. The usual stress test for a GP with one lengthscale per axis. */
export function hartmann6(x) {
  assertDim(x, 'hartmann6', 6);
  const x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4], x5 = x[5];
  let s = 0;
  for (let i = 0, k = 0; i < 4; i++, k += 6) {
    const d0 = x0 - H6_P[k], d1 = x1 - H6_P[k + 1], d2 = x2 - H6_P[k + 2];
    const d3 = x3 - H6_P[k + 3], d4 = x4 - H6_P[k + 4], d5 = x5 - H6_P[k + 5];
    const inner = H6_A[k] * d0 * d0 + H6_A[k + 1] * d1 * d1 + H6_A[k + 2] * d2 * d2
                + H6_A[k + 3] * d3 * d3 + H6_A[k + 4] * d4 * d4 + H6_A[k + 5] * d5 * d5;
    s += H_ALPHA[i] * Math.exp(-inner);
  }
  return -s;
}
hartmann6.meta = {
  name: 'Hartmann-6', dims: 6, sense: 'min', domain: cube(0, 1, 6, 'hartmann6'),
  optimum: {
    // The usual value -3.322368011391339 is f at the usual six-digit
    // location (kept as `approx`), not the minimum, which is 2.4e-11 lower.
    value: -3.3223680114155147,
    at: fixedAt('hartmann6', [
      0.20168951100670543, 0.15001069182345797, 0.476873974221897,
      0.2753324304940561, 0.31165161660011326, 0.6573005340656203,
    ]),
    approx: [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573],
  },
  reference: 'Hartman, J. K. (1973). Naval Research Logistics Quarterly 20(3).',
};

const BRANIN_B = 5.1 / (4 * Math.PI ** 2);
const BRANIN_C = 5 / Math.PI;
const BRANIN_T = 1 / (8 * Math.PI);
const BRANIN_S1MT = 10 * (1 - BRANIN_T);

/**
 * Branin-Hoo. Three equal global minima in a shallow curved trough.
 * Note the ASYMMETRIC domain: x1 in [-5, 10], x2 in [0, 15]. Applying one
 * range to both axes is a common and silent mistake.
 */
export function branin(x, options) {
  assertDim(x, 'branin', 2);
  const x1 = x[0], x2 = x[1];
  if (options === undefined) {
    const u = x2 - BRANIN_B * x1 * x1 + BRANIN_C * x1 - 6;
    return u * u + BRANIN_S1MT * Math.cos(x1) + 10;
  }
  assertOpts(options, 'branin');
  const a = options.a !== undefined ? options.a : 1;
  const b = options.b !== undefined ? options.b : BRANIN_B;
  const c = options.c !== undefined ? options.c : BRANIN_C;
  const r = options.r !== undefined ? options.r : 6;
  const s = options.s !== undefined ? options.s : 10;
  const t = options.t !== undefined ? options.t : BRANIN_T;
  const u = x2 - b * x1 * x1 + c * x1 - r;
  return a * (u * u) + s * (1 - t) * Math.cos(x1) + s;
}
branin.meta = {
  name: 'Branin', dims: 2, sense: 'min',
  domain: (d = 2) => {
    if (d !== 2) throw new RangeError(`branin is 2-D, not ${d}-D`);
    return [[-5, 10], [0, 15]];
  },
  optimum: {
    value: 5 / (4 * Math.PI),
    at: fixedAt('branin', [-Math.PI, 12.275]),
    all: [[-Math.PI, 12.275], [Math.PI, 2.275], [3 * Math.PI, 2.475]],
  },
  reference: 'Branin, F. H. (1972). IBM Journal of Research and Development 16(5).',
};

/** Rosenbrock (1960). The banana valley: easy to enter, slow to follow. */
export function rosenbrock(x) {
  if (!isVec(x)) throw new TypeError('rosenbrock: expected an array of numbers');
  const n = x.length;
  if (n < 2) throw new RangeError(`rosenbrock: unsupported dimension ${n}`);
  let s = 0, xi = x[0];
  for (let i = 0; i < n - 1; i++) {
    const next = x[i + 1];
    const d1 = next - xi * xi;
    const d2 = xi - 1;
    s += 100 * (d1 * d1) + d2 * d2;
    xi = next;
  }
  return s;
}
rosenbrock.meta = {
  name: 'Rosenbrock', dims: '>= 2', sense: 'min', domain: cube(-5, 10, '>= 2', 'rosenbrock'),
  optimum: { value: 0, at: fillAt('rosenbrock', '>= 2', 1) },
  reference: 'Rosenbrock, H. H. (1960). The Computer Journal 3(3).',
};

/** Rastrigin (1974). A regular lattice of local minima on a quadratic bowl. */
export function rastrigin(x) {
  assertVec(x, 'rastrigin');
  const d = x.length;
  let s1 = 0, s2 = 0;
  for (let i = 0; i < d; i++) {
    const xi = x[i];
    s1 += xi * xi;
    s2 += Math.cos(TAU * xi);
  }
  return s1 + 10 * (d - s2);
}
rastrigin.meta = {
  name: 'Rastrigin', dims: 'any', sense: 'min', domain: cube(-5.12, 5.12, 'any', 'rastrigin'),
  optimum: { value: 0, at: fillAt('rastrigin', 'any', 0) },
  reference: 'Rastrigin, L. A. (1974). Systems of Extremal Control.',
};

function powEven(u, m) {
  // u^(2m) by squaring, for positive integer m.
  let p = u * u;
  if (m === 10) {
    const u4 = p * p, u8 = u4 * u4, u16 = u8 * u8;
    return u16 * u4;
  }
  let r = 1, e = m;
  while (e > 0) {
    if (e & 1) r *= p;
    e >>>= 1;
    if (e > 0) p *= p;
  }
  return r;
}

/** Michalewicz (1996). Steep ridges separated by flat plateaus. */
export function michalewicz(x, options) {
  assertVec(x, 'michalewicz');
  let m = 10;
  if (options !== undefined) {
    assertOpts(options, 'michalewicz');
    if (options.m !== undefined) m = options.m;
    if (!(Number.isInteger(m) && m >= 1)) {
      throw new RangeError('michalewicz: m must be a positive integer');
    }
  }
  const d = x.length;
  let s = 0;
  if (m === 10) {
    for (let i = 0; i < d; i++) {
      const xi = x[i];
      const u = Math.sin((i + 1) * xi * xi / Math.PI);
      const u2 = u * u, u4 = u2 * u2, u8 = u4 * u4, u16 = u8 * u8;
      s += Math.sin(xi) * (u16 * u4);
    }
  } else {
    for (let i = 0; i < d; i++) {
      const xi = x[i];
      s += Math.sin(xi) * powEven(Math.sin((i + 1) * xi * xi / Math.PI), m);
    }
  }
  return 0 - s;
}
michalewicz.meta = {
  name: 'Michalewicz', dims: 'any', sense: 'min', domain: cube(0, Math.PI, 'any', 'michalewicz'),
  optimum: {
    value: null,
    // f is a sum of one-dimensional terms, so its d-D minimum is exactly the
    // sum of the 1-D minima. Each was solved at 50 digits; see
    // test/crosscheck_optima.py.
    note: 'depends on d; exact for d = 1..10 at m = 10. Agrees with the published '
        + '-1.8013 (d=2), -4.687658 (d=5) and -9.66015 (d=10).',
    known: {
      1: -0.8013034100985525, 2: -1.8013034100985525, 3: -2.7603946799945587,
      4: -3.698857098466642, 5: -4.687658179088146, 6: -5.687658179088146,
      7: -6.680885314444028, 8: -7.663757350716239, 9: -8.66015171564134,
      10: -9.66015171564134,
    },
  },
  reference: 'Michalewicz, Z. (1996). Genetic Algorithms + Data Structures = Evolution Programs.',
};

/* ------------------------------------------------------------------ *
 * classical psychophysical laws                                      *
 *                                                                    *
 * These are human response functions, not optimization benchmarks.   *
 * They are given in the sense the law states. Three of the four are  *
 * MONOTONE, so on any box their extremum is a corner — which makes   *
 * them weak as search problems but useful as a landscape class that  *
 * the classical benchmarks above do not contain.                     *
 * ------------------------------------------------------------------ */

/** Yerkes-Dodson (1908). Inverted-U of arousal against performance. */
export function yerkesDodson(x, options) {
  assertVec(x, 'yerkesDodson');
  let peak = 0.5, width = 0.15;
  if (options !== undefined) {
    assertOpts(options, 'yerkesDodson');
    if (options.peak !== undefined) peak = options.peak;
    if (options.width !== undefined) width = options.width;
    if (!Number.isFinite(peak)) throw new RangeError('yerkesDodson: peak must be finite');
    if (!(Number.isFinite(width) && width > 0)) {
      throw new RangeError('yerkesDodson: width must be positive');
    }
  }
  const d = x.length, scale = -0.5 / (width * width);
  let s = 0;
  for (let i = 0; i < d; i++) {
    const u = x[i] - peak;
    s += Math.exp(u * u * scale);
  }
  return s / d;
}
yerkesDodson.meta = {
  name: 'Yerkes-Dodson', dims: 'any', sense: 'max', domain: cube(0, 1, 'any', 'yerkesDodson'),
  monotone: false,
  optimum: { value: 1, at: fillAt('yerkesDodson', 'any', 0.5) },
  note: 'The 1908 paper reports the effect, not a formula; the Gaussian inverted-U is a stylisation.',
  reference: 'Yerkes, R. M., & Dodson, J. D. (1908). J. Comparative Neurology and Psychology 18.',
};

/** Stevens' power law (1957): perceived magnitude psi = I^a. Monotone. */
export function stevens(x, options) {
  assertVec(x, 'stevens');
  let a = 0.67;
  if (options !== undefined) {
    assertOpts(options, 'stevens');
    if (options.a !== undefined) a = options.a;
    if (!(Number.isFinite(a) && a > 0)) throw new RangeError('stevens: a must be positive');
  }
  const d = x.length;
  let s = 0;
  for (let i = 0; i < d; i++) {
    const xi = x[i];
    if (xi > 0) s += xi ** a;
    else if (Number.isNaN(xi)) return NaN;
  }
  return s / d;
}
stevens.meta = {
  name: "Stevens' power law", dims: 'any', sense: 'max', domain: cube(0, 1, 'any', 'stevens'),
  monotone: 'increasing',
  optimum: { value: 1, at: fillAt('stevens', 'any', 1) },
  note: 'a = 0.67 is the exponent Stevens reported for loudness.',
  reference: 'Stevens, S. S. (1957). Psychological Review 64(3).',
};

/** Hick-Hyman law (1952/1953): choice reaction time a + b*log2(n + 1). Monotone. */
export function hickHyman(x, options) {
  assertVec(x, 'hickHyman');
  let a = 0.2, b = 0.15;
  if (options !== undefined) {
    assertOpts(options, 'hickHyman');
    if (options.a !== undefined) a = options.a;
    if (options.b !== undefined) b = options.b;
    if (!Number.isFinite(a)) throw new RangeError('hickHyman: a must be finite');
    if (!(Number.isFinite(b) && b > 0)) throw new RangeError('hickHyman: b must be positive');
  }
  const d = x.length;
  let s = 0;
  for (let i = 0; i < d; i++) {
    s += Math.log2(Math.max(x[i], 0) + 1);
  }
  return a + b * (s / d);
}
hickHyman.meta = {
  name: 'Hick-Hyman law', dims: 'any', sense: 'min', domain: cube(0, 10, 'any', 'hickHyman'),
  monotone: 'increasing',
  optimum: { value: 0.2, at: fillAt('hickHyman', 'any', 0) },
  note: 'Returns reaction time in seconds, so lower is better. a and b are in the usual reported range.',
  reference: 'Hick, W. E. (1952). QJEP 4(1); Hyman, R. (1953). J. Experimental Psychology 45(3).',
};

/** Weber-Fechner (1860): sensation psi = ln(1 + I / I0). Monotone. */
export function weberFechner(x, options) {
  assertVec(x, 'weberFechner');
  let i0 = 0.01;
  if (options !== undefined) {
    assertOpts(options, 'weberFechner');
    if (options.i0 !== undefined) i0 = options.i0;
    if (!(Number.isFinite(i0) && i0 > 0)) throw new RangeError('weberFechner: i0 must be positive');
  }
  const d = x.length;
  let s = 0;
  for (let i = 0; i < d; i++) s += Math.log1p(Math.max(x[i], 0) / i0);
  return s / d;
}
weberFechner.meta = {
  name: 'Weber-Fechner', dims: 'any', sense: 'max', domain: cube(0, 1, 'any', 'weberFechner'),
  monotone: 'increasing',
  optimum: { value: Math.log1p(1 / 0.01), at: fillAt('weberFechner', 'any', 1) },
  note: 'ln(1 + I/I0) rather than ln(I/I0), so the value stays finite at zero intensity.',
  reference: 'Fechner, G. T. (1860). Elemente der Psychophysik.',
};

/* ------------------------------------------------------------------ *
 * registry                                                           *
 * ------------------------------------------------------------------ */

export const functions = {
  ackley, griewank, schwefel, eggholder, powell, shekel,
  hartmann3, hartmann6, branin, rosenbrock, rastrigin, michalewicz,
  yerkesDodson, stevens, hickHyman, weberFechner,
};

/** Every benchmark that is a classical optimization test function. */
export const optimizationBenchmarks = [
  'ackley', 'griewank', 'schwefel', 'eggholder', 'powell', 'shekel',
  'hartmann3', 'hartmann6', 'branin', 'rosenbrock', 'rastrigin', 'michalewicz',
];

/** Every benchmark that is a classical human response function. */
export const psychophysicalLaws = [
  'yerkesDodson', 'stevens', 'hickHyman', 'weberFechner',
];

const CUBE_BOUNDS = {
  ackley: [-32.768, 32.768], griewank: [-600, 600], schwefel: [-500, 500],
  eggholder: [-512, 512], powell: [-4, 5], shekel: [0, 10],
  hartmann3: [0, 1], hartmann6: [0, 1], rosenbrock: [-5, 10],
  rastrigin: [-5.12, 5.12], michalewicz: [0, Math.PI],
  yerkesDodson: [0, 1], stevens: [0, 1], hickHyman: [0, 10], weberFechner: [0, 1],
};

/** The per-axis domain of `name` at dimension `d`, as [[lo, hi], ...]. */
export function domain(name, d) {
  if (!Object.hasOwn(functions, name)) throw new RangeError(`unknown function: ${name}`);
  const f = functions[name];
  return f.meta.domain(checkDimFor(name, f.meta.dims, d));
}

/** Uniformly map a point from the unit cube onto `name`'s own domain. */
export function fromUnitCube(name, u) {
  if (!Object.hasOwn(functions, name)) throw new RangeError(`unknown function: ${name}`);
  if (!isVec(u)) throw new TypeError('fromUnitCube: expected an array of numbers');
  const f = functions[name];
  const d = checkDimFor(name, f.meta.dims, u.length);
  // Plain Array, not u.map: a Float32Array input would otherwise come back as
  // a Float32Array and round the mapped point to single precision.
  if (name === 'branin') return [-5 + u[0] * 15, u[1] * 15];
  const bounds = Object.hasOwn(CUBE_BOUNDS, name) ? CUBE_BOUNDS[name] : null;
  if (bounds) {
    const lo = bounds[0], span = bounds[1] - lo;
    const out = new Array(d);
    for (let i = 0; i < d; i++) out[i] = lo + u[i] * span;
    return out;
  }
  const box = f.meta.domain(d);
  const out = new Array(d);
  for (let i = 0; i < d; i++) out[i] = box[i][0] + u[i] * (box[i][1] - box[i][0]);
  return out;
}

export default functions;
