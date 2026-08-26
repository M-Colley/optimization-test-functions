# -*- coding: utf-8 -*-
"""Cross-check this library against BoTorch's synthetic test functions.

Optional developer check. BoTorch is the reference implementation most of the
Bayesian-optimization field uses, so agreeing with it value-for-value is
stronger evidence than any amount of re-reading the formulas.

    pip install botorch
    python test/crosscheck_botorch.py

Not all of our functions exist in BoTorch (Schwefel and the psychophysical laws
do not); those are skipped and reported as such.
"""
import json
import math
import os
import subprocess
import sys

import torch

# BoTorch stores ALPHA/A/C as float32 and upcasts, so its own constants carry
# ~1e-8 of rounding. Ours are exact float64. Compare at 1e-6.
TOL = 1e-6
import botorch.test_functions.synthetic as syn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ours -> (BoTorch class, dimension). BoTorch defaults to minimization
# (negate=False), which is the convention this library uses too.
PAIRS = {
    "ackley":      ("Ackley", 4),
    "griewank":    ("Griewank", 4),
    "eggholder":   ("EggHolder", 2),
    "powell":      ("Powell", 4),
    "shekel":      ("Shekel", 4),
    "hartmann3":   ("Hartmann", 3),
    "hartmann6":   ("Hartmann", 6),
    "branin":      ("Branin", 2),
    "rosenbrock":  ("Rosenbrock", 4),
    "rastrigin":   ("Rastrigin", 4),
    "michalewicz": ("Michalewicz", 5),
}
SKIP = {
    "schwefel": "not in BoTorch",
    "yerkesDodson": "psychophysical law, not an optimization benchmark",
    "stevens": "psychophysical law, not an optimization benchmark",
    "hickHyman": "psychophysical law, not an optimization benchmark",
    "weberFechner": "psychophysical law, not an optimization benchmark",
}


def build(name, dim):
    cls = getattr(syn, name)
    try:
        return cls(dim=dim).to(dtype=torch.double)
    except TypeError:                       # fixed-dimension problems
        p = cls().to(dtype=torch.double)
        if p.dim != dim:
            raise ValueError("%s is %d-D, expected %d" % (name, p.dim, dim))
        return p


def points(box, n=25):
    """A deterministic spread over the box, plus its corners and centre."""
    d = len(box)
    out = []
    for k in range(n):
        row = []
        for j in range(d):
            u = ((k + 1) * (j + 2) * 0.6180339887498949) % 1.0
            row.append(box[j][0] + u * (box[j][1] - box[j][0]))
        out.append(row)
    out.append([b[0] for b in box])
    out.append([b[1] for b in box])
    out.append([(b[0] + b[1]) / 2 for b in box])
    return out


# ask the JS side for its domains and values ------------------------------
cases = {}
for ours, (_bt, dim) in PAIRS.items():
    cases[ours] = {"dim": dim}
probe = os.path.join(HERE, "_probe.mjs")
with open(probe, "w", encoding="utf-8") as fh:
    fh.write("""
import { functions, domain } from '../src/index.js';
const spec = JSON.parse(process.argv[2]);
const out = {};
for (const [name, { dim }] of Object.entries(spec)) {
  out[name] = { box: domain(name, dim) };
}
process.stdout.write(JSON.stringify(out));
""")
r = subprocess.run(["node", probe, json.dumps(cases)], capture_output=True, text=True, cwd=HERE)
if r.returncode != 0:
    os.remove(probe)
    sys.exit("node probe failed:\n" + r.stdout + r.stderr)
boxes = {k: v["box"] for k, v in json.loads(r.stdout).items()}

allcases = {name: points(boxes[name]) for name in PAIRS}
with open(probe, "w", encoding="utf-8") as fh:
    fh.write("""
import { functions } from '../src/index.js';
const cases = JSON.parse(process.argv[2]);
const out = {};
for (const [name, pts] of Object.entries(cases)) out[name] = pts.map((x) => functions[name](x));
process.stdout.write(JSON.stringify(out));
""")
r = subprocess.run(["node", probe, json.dumps(allcases)], capture_output=True, text=True, cwd=HERE)
os.remove(probe)
if r.returncode != 0:
    sys.exit("node eval failed:\n" + r.stdout + r.stderr)
ours_vals = json.loads(r.stdout)

# compare ------------------------------------------------------------------
fails, checks = [], 0
print("%-14s %-16s %s" % ("function", "botorch", "result"))
print("-" * 52)
for name, (bt, dim) in PAIRS.items():
    prob = build(bt, dim)
    bad = 0
    for i, x in enumerate(allcases[name]):
        checks += 1
        theirs = float(prob(torch.tensor([x], dtype=torch.double), noise=False).reshape(()))
        mine = ours_vals[name][i]
        scale = max(1.0, abs(theirs))
        if not math.isfinite(mine) or abs(theirs - mine) / scale > TOL:
            bad += 1
            if len(fails) < 12:
                fails.append("%-12s pt%-3d botorch=%.12g  ours=%.12g" % (name, i, theirs, mine))
    print("%-14s %-16s %s" % (name, bt, "ok" if bad == 0 else "%d MISMATCH" % bad))

for name, why in SKIP.items():
    print("%-14s %-16s skipped (%s)" % (name, "-", why))

print("\n%d value(s) compared across %d function(s)" % (checks, len(PAIRS)))
if fails:
    print("\nMISMATCHES:")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("all agree with BoTorch to %g relative" % TOL)
