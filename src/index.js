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

function assertLength(x, name, ok) {
  if (!Array.isArray(x) && !ArrayBuffer.isView(x)) {
    throw new TypeError(`${name}: expected an array of numbers`);
  }
  if (!ok(x.length)) {
    throw new RangeError(`${name}: unsupported dimension ${x.length}`);
  }
}

const mean = (x) => {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i];
  return s / x.length;
};

/** Wrap a function so a minimizer can be used as a maximizer, or vice versa. */
export function negate(f) {
  const g = (...args) => -f(...args);
  if (f.meta) g.meta = { ...f.meta, sense: f.meta.sense === 'min' ? 'max' : 'min' };
  return g;
}

/** Expand a per-axis domain to `d` dimensions, for functions defined on a cube. */
function cube(lo, hi) {
  return (d) => Array.from({ length: d }, () => [lo, hi]);
}

/* ------------------------------------------------------------------ *
 * classical optimization benchmarks (minimized)                      *
 * ------------------------------------------------------------------ */

/** Ackley (1987). Flat outer plain, narrow basin at the origin. */
export function ackley(x, { a = 20, b = 0.2, c = TAU } = {}) {
  assertLength(x, 'ackley', (n) => n >= 1);
  const d = x.length;
  let s1 = 0, s2 = 0;
  for (let i = 0; i < d; i++) { s1 += x[i] * x[i]; s2 += Math.cos(c * x[i]); }
  return -a * Math.exp(-b * Math.sqrt(s1 / d)) - Math.exp(s2 / d) + a + Math.E;
}
ackley.meta = {
  name: 'Ackley', dims: 'any', sense: 'min', domain: cube(-32.768, 32.768),
  optimum: { value: 0, at: (d) => new Array(d).fill(0) },
  reference: 'Ackley, D. H. (1987). A Connectionist Machine for Genetic Hillclimbing.',
};

/** Griewank (1981). Wide bowl with fine cosine ripples. */
export function griewank(x) {
  assertLength(x, 'griewank', (n) => n >= 1);
  let s = 0, p = 1;
  for (let i = 0; i < x.length; i++) {
    s += x[i] * x[i];
    p *= Math.cos(x[i] / Math.sqrt(i + 1));
  }
  return s / 4000 - p + 1;
}
griewank.meta = {
  name: 'Griewank', dims: 'any', sense: 'min', domain: cube(-600, 600),
  optimum: { value: 0, at: (d) => new Array(d).fill(0) },
  reference: 'Griewank, A. O. (1981). Journal of Optimization Theory and Applications 34(1).',
};

/** Schwefel (1981). Global optimum far from the deepest local one. */
export function schwefel(x) {
  assertLength(x, 'schwefel', (n) => n >= 1);
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * Math.sin(Math.sqrt(Math.abs(x[i])));
  return 418.9828872724339 * x.length - s;
}
schwefel.meta = {
  name: 'Schwefel', dims: 'any', sense: 'min', domain: cube(-500, 500),
  optimum: { value: 0, at: (d) => new Array(d).fill(420.9687463) },
  reference: 'Schwefel, H.-P. (1981). Numerical Optimization of Computer Models.',
};

/** Eggholder. Two dimensions, wildly multimodal, no usable global structure. */
export function eggholder(x) {
  assertLength(x, 'eggholder', (n) => n === 2);
  const [a, b] = x;
  return -(b + 47) * Math.sin(Math.sqrt(Math.abs(a / 2 + (b + 47))))
         - a * Math.sin(Math.sqrt(Math.abs(a - (b + 47))));
}
eggholder.meta = {
  name: 'Eggholder', dims: 2, sense: 'min', domain: () => [[-512, 512], [-512, 512]],
  optimum: { value: -959.6406627208506, at: () => [512, 404.2318058008512] },
  reference: 'Vanaret et al. (2014); after Whitley et al. (1996).',
};

/** Powell (1962). Smooth quartic valley. Dimension must be a multiple of 4. */
export function powell(x) {
  assertLength(x, 'powell', (n) => n > 0 && n % 4 === 0);
  let s = 0;
  for (let i = 0; i < x.length; i += 4) {
    s += (x[i] + 10 * x[i + 1]) ** 2
       + 5 * (x[i + 2] - x[i + 3]) ** 2
       + (x[i + 1] - 2 * x[i + 2]) ** 4
       + 10 * (x[i] - x[i + 3]) ** 4;
  }
  return s;
}
powell.meta = {
  name: 'Powell', dims: 'multiple of 4', sense: 'min', domain: cube(-4, 5),
  optimum: { value: 0, at: (d) => new Array(d).fill(0) },
  reference: 'Powell, M. J. D. (1962). The Computer Journal 5(2).',
};

const SHEKEL_A = [
  [4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
  [2, 9, 2, 9], [5, 3, 5, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6],
];
const SHEKEL_C = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5];

/** Shekel. Four dimensions, `m` narrow wells on a flat plain. */
export function shekel(x, { m = 10 } = {}) {
  assertLength(x, 'shekel', (n) => n === 4);
  if (!(m >= 1 && m <= 10)) throw new RangeError('shekel: m must be 1..10');
  let s = 0;
  for (let i = 0; i < m; i++) {
    let q = 0;
    for (let j = 0; j < 4; j++) { const d = x[j] - SHEKEL_A[i][j]; q += d * d; }
    s += 1 / (q + SHEKEL_C[i]);
  }
  return -s;
}
shekel.meta = {
  name: 'Shekel', dims: 4, sense: 'min', domain: cube(0, 10),
  optimum: {
    value: -10.53644315348353,
    // Almost always quoted as (4, 4, 4, 4), which is only approximate: the
    // neighbouring wells pull the minimum slightly off that corner, and
    // f(4,4,4,4) = -10.536283, not -10.536443. Refined location below.
    at: () => [4.0007468671, 3.9995094806, 4.0007468700, 3.9995094776],
    approx: [4, 4, 4, 4],
    note: 'for m = 10',
  },
  reference: 'Shekel, J. (1971). Proc. 1971 IEEE Conf. on Decision and Control.',
};

const H_ALPHA = [1.0, 1.2, 3.0, 3.2];
const H3_A = [[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]];
const H3_P = [[3689, 1170, 2673], [4699, 4387, 7470],
              [1091, 8732, 5547], [381, 5743, 8828]].map((r) => r.map((v) => v * 1e-4));

/** Hartmann 3-D. Four Gaussian basins of unequal depth. */
export function hartmann3(x) {
  assertLength(x, 'hartmann3', (n) => n === 3);
  let s = 0;
  for (let i = 0; i < 4; i++) {
    let inner = 0;
    for (let j = 0; j < 3; j++) { const d = x[j] - H3_P[i][j]; inner += H3_A[i][j] * d * d; }
    s += H_ALPHA[i] * Math.exp(-inner);
  }
  return -s;
}
hartmann3.meta = {
  name: 'Hartmann-3', dims: 3, sense: 'min', domain: cube(0, 1),
  optimum: { value: -3.86278214782076, at: () => [0.114614, 0.555649, 0.852547] },
  reference: 'Hartman, J. K. (1973). Naval Research Logistics Quarterly 20(3).',
};

const H6_A = [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
              [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]];
const H6_P = [[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
              [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]]
             .map((r) => r.map((v) => v * 1e-4));

/** Hartmann 6-D. The usual stress test for a GP with one lengthscale per axis. */
export function hartmann6(x) {
  assertLength(x, 'hartmann6', (n) => n === 6);
  let s = 0;
  for (let i = 0; i < 4; i++) {
    let inner = 0;
    for (let j = 0; j < 6; j++) { const d = x[j] - H6_P[i][j]; inner += H6_A[i][j] * d * d; }
    s += H_ALPHA[i] * Math.exp(-inner);
  }
  return -s;
}
hartmann6.meta = {
  name: 'Hartmann-6', dims: 6, sense: 'min', domain: cube(0, 1),
  optimum: {
    value: -3.322368011391339,
    at: () => [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573],
  },
  reference: 'Hartman, J. K. (1973). Naval Research Logistics Quarterly 20(3).',
};

/**
 * Branin-Hoo. Three equal global minima in a shallow curved trough.
 * Note the ASYMMETRIC domain: x1 in [-5, 10], x2 in [0, 15]. Applying one
 * range to both axes is a common and silent mistake.
 */
export function branin(x, { a = 1, b = 5.1 / (4 * Math.PI ** 2), c = 5 / Math.PI,
                            r = 6, s = 10, t = 1 / (8 * Math.PI) } = {}) {
  assertLength(x, 'branin', (n) => n === 2);
  const [x1, x2] = x;
  return a * (x2 - b * x1 * x1 + c * x1 - r) ** 2 + s * (1 - t) * Math.cos(x1) + s;
}
branin.meta = {
  name: 'Branin', dims: 2, sense: 'min', domain: () => [[-5, 10], [0, 15]],
  optimum: {
    value: 0.39788735772973816,
    at: () => [-Math.PI, 12.275],
    all: [[-Math.PI, 12.275], [Math.PI, 2.275], [9.42478, 2.475]],
  },
  reference: 'Branin, F. H. (1972). IBM Journal of Research and Development 16(5).',
};

/** Rosenbrock (1960). The banana valley: easy to enter, slow to follow. */
export function rosenbrock(x) {
  assertLength(x, 'rosenbrock', (n) => n >= 2);
  let s = 0;
  for (let i = 0; i < x.length - 1; i++) {
    s += 100 * (x[i + 1] - x[i] * x[i]) ** 2 + (x[i] - 1) ** 2;
  }
  return s;
}
rosenbrock.meta = {
  name: 'Rosenbrock', dims: '>= 2', sense: 'min', domain: cube(-5, 10),
  optimum: { value: 0, at: (d) => new Array(d).fill(1) },
  reference: 'Rosenbrock, H. H. (1960). The Computer Journal 3(3).',
};

/** Rastrigin (1974). A regular lattice of local minima on a quadratic bowl. */
export function rastrigin(x) {
  assertLength(x, 'rastrigin', (n) => n >= 1);
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i] - 10 * Math.cos(TAU * x[i]);
  return 10 * x.length + s;
}
rastrigin.meta = {
  name: 'Rastrigin', dims: 'any', sense: 'min', domain: cube(-5.12, 5.12),
  optimum: { value: 0, at: (d) => new Array(d).fill(0) },
  reference: 'Rastrigin, L. A. (1974). Systems of Extremal Control.',
};

/** Michalewicz (1996). Steep ridges separated by flat plateaus. */
export function michalewicz(x, { m = 10 } = {}) {
  assertLength(x, 'michalewicz', (n) => n >= 1);
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    s += Math.sin(x[i]) * Math.sin((i + 1) * x[i] * x[i] / Math.PI) ** (2 * m);
  }
  return -s;
}
michalewicz.meta = {
  name: 'Michalewicz', dims: 'any', sense: 'min', domain: cube(0, Math.PI),
  optimum: {
    value: null,
    note: 'depends on d; published values are -1.8013 (d=2), -4.687658 (d=5), -9.66015 (d=10) at m=10',
    known: { 2: -1.8013, 5: -4.687658, 10: -9.66015 },
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
export function yerkesDodson(x, { peak = 0.5, width = 0.15 } = {}) {
  assertLength(x, 'yerkesDodson', (n) => n >= 1);
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    s += Math.exp(-((x[i] - peak) ** 2) / (2 * width * width));
  }
  return s / x.length;
}
yerkesDodson.meta = {
  name: 'Yerkes-Dodson', dims: 'any', sense: 'max', domain: cube(0, 1), monotone: false,
  optimum: { value: 1, at: (d) => new Array(d).fill(0.5) },
  reference: 'Yerkes, R. M., & Dodson, J. D. (1908). J. Comparative Neurology and Psychology 18.',
};

/** Stevens' power law (1957): perceived magnitude psi = I^a. Monotone. */
export function stevens(x, { a = 0.67 } = {}) {
  assertLength(x, 'stevens', (n) => n >= 1);
  let s = 0;
  for (let i = 0; i < x.length; i++) s += Math.max(x[i], 0) ** a;
  return s / x.length;
}
stevens.meta = {
  name: "Stevens' power law", dims: 'any', sense: 'max', domain: cube(0, 1),
  monotone: 'increasing',
  optimum: { value: 1, at: (d) => new Array(d).fill(1) },
  note: 'a = 0.67 is the exponent Stevens reported for loudness.',
  reference: 'Stevens, S. S. (1957). Psychological Review 64(3).',
};

/** Hick-Hyman law (1952/1953): choice reaction time a + b*log2(n + 1). Monotone. */
export function hickHyman(x, { a = 0.2, b = 0.15 } = {}) {
  assertLength(x, 'hickHyman', (n) => n >= 1);
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    s += a + b * Math.log2(Math.max(x[i], 0) + 1);
  }
  return s / x.length;
}
hickHyman.meta = {
  name: "Hick-Hyman law", dims: 'any', sense: 'min', domain: cube(0, 10),
  monotone: 'increasing',
  optimum: { value: 0.2, at: (d) => new Array(d).fill(0) },
  note: 'Returns reaction time in seconds, so lower is better. a and b are in the usual reported range.',
  reference: 'Hick, W. E. (1952). QJEP 4(1); Hyman, R. (1953). J. Experimental Psychology 45(3).',
};

/** Weber-Fechner (1860): sensation psi = ln(1 + I / I0). Monotone. */
export function weberFechner(x, { i0 = 0.01 } = {}) {
  assertLength(x, 'weberFechner', (n) => n >= 1);
  let s = 0;
  for (let i = 0; i < x.length; i++) s += Math.log1p(Math.max(x[i], 0) / i0);
  return s / x.length;
}
weberFechner.meta = {
  name: 'Weber-Fechner', dims: 'any', sense: 'max', domain: cube(0, 1),
  monotone: 'increasing',
  optimum: { value: Math.log1p(1 / 0.01), at: (d) => new Array(d).fill(1) },
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

/** The per-axis domain of `name` at dimension `d`, as [[lo, hi], ...]. */
export function domain(name, d) {
  const f = functions[name];
  if (!f) throw new RangeError(`unknown function: ${name}`);
  const dims = f.meta.dims;
  if (typeof dims === 'number') d = dims;
  if (d === undefined) throw new RangeError(`${name}: dimension is required`);
  return f.meta.domain(d);
}

/** Uniformly map a point from the unit cube onto `name`'s own domain. */
export function fromUnitCube(name, u) {
  const box = domain(name, u.length);
  return u.map((v, i) => box[i][0] + v * (box[i][1] - box[i][0]));
}

export default functions;
