# optimization-test-functions

[![npm](https://img.shields.io/npm/v/optimization-test-functions.svg)](https://www.npmjs.com/package/optimization-test-functions)
[![CI](https://github.com/M-Colley/optimization-test-functions/actions/workflows/ci.yml/badge.svg)](https://github.com/M-Colley/optimization-test-functions/actions/workflows/ci.yml)
[![license](https://img.shields.io/npm/l/optimization-test-functions.svg)](./LICENSE)

Standard global-optimization benchmarks and classical psychophysical laws, in
JavaScript. No dependencies, no build step, ESM.

**On npm:** <https://www.npmjs.com/package/optimization-test-functions>

Every function ships with the metadata you need to use it correctly: the domain
it is actually defined on, its known optimum, and which direction is the
interesting one.

```bash
npm install optimization-test-functions
```

```js
import { ackley, branin, domain, negate } from 'optimization-test-functions';

ackley([0, 0, 0]);            // 0        — the global minimum
branin([-Math.PI, 12.275]);   // 0.397887 — one of three equal minima

domain('branin');             // [[-5, 10], [0, 15]]  — note the asymmetry
domain('ackley', 4);          // four copies of [-32.768, 32.768]

const maximise = negate(ackley);
maximise.meta.sense;          // 'max'
```

## Why this exists

There was no JavaScript implementation of these. Python has BoTorch, DEAP and
several standalone packages; the npm registry had nothing, so anyone writing an
optimizer for the browser was retyping Hartmann's constant tables by hand.

Two things this gets right that hand-ported versions usually do not:

**Domains are per-axis.** Branin is defined on `x₁ ∈ [-5, 10], x₂ ∈ [0, 15]`.
Applying one range to both axes is a common and completely silent mistake — the
function still evaluates, it just is not Branin any more.

**Nothing is silently negated.** The optimization benchmarks are minimized, as
their sources define them. The psychophysical laws are given in the sense the
law states — Hick–Hyman returns a reaction time, so lower is better, and
`meta.sense` says so. If your optimizer maximizes, wrap with `negate`.

## What's included

### Classical optimization benchmarks — minimized

| function | dims | domain | minimum |
|---|---|---|---|
| `ackley` | any | `[-32.768, 32.768]ᵈ` | `0` at the origin |
| `griewank` | any | `[-600, 600]ᵈ` | `0` at the origin |
| `schwefel` | any | `[-500, 500]ᵈ` | `0` at `(420.9687, …)` |
| `eggholder` | 2 | `[-512, 512]²` | `-959.6407` |
| `powell` | multiple of 4 | `[-4, 5]ᵈ` | `0` at the origin |
| `shekel` | 4 | `[0, 10]⁴` | `-10.5364` |
| `hartmann3` | 3 | `[0, 1]³` | `-3.86278` |
| `hartmann6` | 6 | `[0, 1]⁶` | `-3.32237` |
| `branin` | 2 | `[-5, 10] × [0, 15]` | `0.397887`, three times |
| `rosenbrock` | ≥ 2 | `[-5, 10]ᵈ` | `0` at all-ones |
| `rastrigin` | any | `[-5.12, 5.12]ᵈ` | `0` at the origin |
| `michalewicz` | any | `[0, π]ᵈ` | d-dependent |

### Classical psychophysical laws

Human response functions rather than optimization benchmarks. Three of the four
are monotone, so on any box their extremum sits on the boundary — weak as search
problems, but a landscape class the benchmarks above do not contain, and one
some acquisition functions handle badly.

| function | law | sense |
|---|---|---|
| `yerkesDodson` | inverted-U of arousal against performance (1908) | max, interior |
| `stevens` | ψ = I^0.67, Stevens' power law (1957) | max, boundary |
| `hickHyman` | RT = a + b·log₂(n+1), Hick (1952) / Hyman (1953) | min, boundary |
| `weberFechner` | ψ = ln(1 + I/I₀), Fechner (1860) | max, boundary |

## Metadata

```js
import { shekel } from 'optimization-test-functions';

shekel.meta;
// {
//   name: 'Shekel', dims: 4, sense: 'min',
//   domain: d => [[0, 10], [0, 10], [0, 10], [0, 10]],
//   optimum: {
//     value: -10.53644315348353,
//     at: () => [4.0007468671, 3.9995094806, 4.00074687, 3.9995094776],
//     approx: [4, 4, 4, 4],
//     note: 'for m = 10',
//   },
//   reference: 'Shekel, J. (1971). …',
// }
```

Shekel is a good example of why the metadata is worth having. Its minimum is
almost always quoted at `(4, 4, 4, 4)`, but that is only approximate — the
neighbouring wells pull it slightly off, and `f(4,4,4,4) = -10.536283`, not
`-10.536443`. Both the quoted and the refined location are recorded.

## Optional parameters

```js
shekel(x, { m: 5 });                  // fewer wells
michalewicz(x, { m: 1 });             // gentler ridges (default 10)
ackley(x, { a: 20, b: 0.2, c: 2 * Math.PI });
branin(x, { a: 1, b: 5.1 / (4 * Math.PI ** 2), /* … */ });
yerkesDodson(x, { peak: 0.5, width: 0.15 });
hickHyman(x, { a: 0.2, b: 0.15 });
```

## Helpers

```js
import { functions, optimizationBenchmarks, psychophysicalLaws,
         domain, fromUnitCube, negate } from 'optimization-test-functions';

functions.ackley([1, 2]);                  // registry, keyed by name
optimizationBenchmarks;                    // the 12 classical names
psychophysicalLaws;                        // the 4 human response names
fromUnitCube('branin', [0.5, 0.5]);        // [2.5, 7.5]
```

`fromUnitCube` matters more than it looks: most optimizers work on the unit
cube, and mapping back with a single shared range is exactly how Branin gets
evaluated on the wrong box.

## Correctness

Two independent checks, both run in CI:

- `npm test` — 11 suites. Verifies that every stated optimum is actually
  attained at its stated location, and that 200 000 random samples inside each
  domain fail to beat it. Also covers dimension validation, `negate`, and the
  per-axis domain mapping.
- `python test/crosscheck_botorch.py` — compares **308 values across 11
  functions** against [BoTorch](https://botorch.org)'s `test_functions.synthetic`,
  the reference implementation most of the Bayesian-optimization field uses.

  | function | max relative difference |
  |---|---|
  | `ackley`, `eggholder`, `powell`, `rosenbrock`, `rastrigin` | `0` exactly |
  | `griewank` | `3.5e-16` |
  | `branin` | `5.5e-16` |
  | `michalewicz` | `2.2e-15` |
  | `shekel` | `5.8e-09` |
  | `hartmann6` | `1.2e-08` |
  | `hartmann3` | `2.9e-08` |

  Nothing exceeds `3e-08`, and everything above machine epsilon is BoTorch's
  own doing: it stores `ALPHA`, `A` and `C` as **float32** and upcasts, so its
  constants carry about `1e-8` of rounding while the values here are exact
  float64. That is why the tolerance is `1e-6` and not tighter — tightening it
  fails on BoTorch's precision, not on ours.

Schwefel is not in BoTorch, and the psychophysical laws are not optimization
benchmarks, so those five are checked by the JS suite only.

## Notes

ESM only. Node 18+. For CommonJS, use a dynamic import:

```js
const { ackley } = await import('optimization-test-functions');
```

TypeScript types are hand-written and shipped alongside — there is no build
step, so what you install is what is in `src/`.

## License

MIT
