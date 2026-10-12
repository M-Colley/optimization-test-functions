export type Vec = readonly number[] | Float64Array | Float32Array;

/** One [lo, hi] pair per axis. */
export type Box = [number, number][];

export interface Optimum {
  /** The optimal value, or null when it depends on the dimension. */
  value: number | null;
  /** The location, given the dimension. Absent when `value` is null. */
  at?: (d: number) => number[];
  /** A commonly quoted but only approximate location, where one exists. */
  approx?: number[];
  /** Every global optimum, for functions that have more than one. */
  all?: number[][];
  /** Optimal values keyed by dimension, when the optimum is d-dependent. */
  known?: Record<number, number>;
  note?: string;
}

export interface Meta {
  name: string;
  /** 'any', a fixed number, or a description such as 'multiple of 4'. */
  dims: number | string;
  /** Which direction is interesting. Nothing here is silently negated. */
  sense: 'min' | 'max';
  /** The function's own domain, per axis, at dimension d. */
  domain: (d: number) => Box;
  optimum: Optimum;
  /** Present on the psychophysical laws; false or a direction. */
  monotone?: false | 'increasing' | 'decreasing';
  note?: string;
  reference: string;
}

export interface TestFunction<O = Record<string, never>> {
  (x: Vec, options?: O): number;
  meta: Meta;
}

/* classical optimization benchmarks -- minimized */
export const ackley: TestFunction<{ a?: number; b?: number; c?: number }>;
export const griewank: TestFunction;
export const schwefel: TestFunction;
export const eggholder: TestFunction;
export const powell: TestFunction;
export const shekel: TestFunction<{ m?: number }>;
export const hartmann3: TestFunction;
export const hartmann6: TestFunction;
export const branin: TestFunction<{
  a?: number; b?: number; c?: number; r?: number; s?: number; t?: number;
}>;
export const rosenbrock: TestFunction;
export const rastrigin: TestFunction;
export const michalewicz: TestFunction<{ m?: number }>;

/* classical psychophysical laws */
export const yerkesDodson: TestFunction<{ peak?: number; width?: number }>;
export const stevens: TestFunction<{ a?: number }>;
export const hickHyman: TestFunction<{ a?: number; b?: number }>;
export const weberFechner: TestFunction<{ i0?: number }>;

export type FunctionName =
  | 'ackley' | 'griewank' | 'schwefel' | 'eggholder' | 'powell' | 'shekel'
  | 'hartmann3' | 'hartmann6' | 'branin' | 'rosenbrock' | 'rastrigin'
  | 'michalewicz' | 'yerkesDodson' | 'stevens' | 'hickHyman' | 'weberFechner';

export const functions: Record<FunctionName, TestFunction<any>>;
export const optimizationBenchmarks: FunctionName[];
export const psychophysicalLaws: FunctionName[];

/**
 * Wrap a function so a minimizer can be used as a maximizer, or vice versa.
 * `meta.sense`, `meta.optimum.value`, `meta.optimum.known` and
 * `meta.monotone` all flip with the sign.
 */
export function negate<O>(f: TestFunction<O>): TestFunction<O>;

/**
 * The per-axis domain of `name` at dimension `d`. `d` is optional for
 * fixed-dimension functions and must match their dimension if given.
 */
export function domain(name: FunctionName, d?: number): Box;

/** Map a point from the unit cube onto `name`'s own domain. */
export function fromUnitCube(name: FunctionName, u: Vec): number[];

declare const _default: Record<FunctionName, TestFunction<any>>;
export default _default;
