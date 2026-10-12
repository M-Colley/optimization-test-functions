# -*- coding: utf-8 -*-
"""Re-derive every stated optimum at 50 significant digits and compare.

Optional developer check, the companion to crosscheck_botorch.py. That script
checks the formulas; this one checks the metadata. The literature quotes most
optima to five or six digits, and some widely copied values are not attained
by the function they are quoted for (Hartmann-3's -3.86278214782076 is 2.4e-6
below its true minimum). So each optimum here is re-derived from scratch with
mpmath, using the exact decimal constants rather than their float64 images:

  * interior optima: Newton on grad f = 0 from the stated location;
  * Eggholder: the same, restricted to the x1 = 512 face it sits on, plus a
    check that df/dx1 < 0 there so the face really is where the minimum is;
  * Schwefel: tan(s) = -s/2 with x = s^2, the 1-D stationarity condition;
  * Michalewicz: f is a sum of 1-D terms, so its d-D minimum is the sum of
    the 1-D minima. Each is bracketed on a dense grid and solved exactly,
    which gives the optimum for every d, not only the three usually quoted.

    pip install mpmath numpy
    python test/crosscheck_optima.py
"""
import json
import os
import subprocess
import sys

import mpmath as mp
import numpy as np

mp.mp.dps = 50
pi = mp.pi
HERE = os.path.dirname(os.path.abspath(__file__))

# Stated values must agree to a few float64 ulps; locations to 1e-9, which is
# far tighter than any optimum is usually quoted and loose enough for flat
# minima, where the location is less well determined than the value.
VALUE_TOL = 1e-14
AT_TOL = 1e-9


def newton(grad, x0):
    return mp.findroot(grad, [mp.mpf(v) for v in x0], tol=mp.mpf(10) ** -40, maxsteps=200)


# ask the JS side for its stated optima ------------------------------------
probe = os.path.join(HERE, "_probe.mjs")
with open(probe, "w", encoding="utf-8") as fh:
    fh.write("""
import { functions } from '../src/index.js';
const dims = { schwefel: 1, eggholder: 2, shekel: 4, hartmann3: 3, hartmann6: 6, branin: 2 };
const out = {};
for (const [name, d] of Object.entries(dims)) {
  const o = functions[name].meta.optimum;
  out[name] = { value: o.value, at: o.at(d), all: o.all ?? null };
}
out.michalewicz = { known: functions.michalewicz.meta.optimum.known };
process.stdout.write(JSON.stringify(out));
""")
r = subprocess.run(["node", probe], capture_output=True, text=True, cwd=HERE)
os.remove(probe)
if r.returncode != 0:
    sys.exit("node probe failed:\n" + r.stdout + r.stderr)
stated = json.loads(r.stdout)

# derive -------------------------------------------------------------------
derived = {}

# Schwefel: d/dx [x sin sqrt x] = 0  <=>  tan(s) = -s/2,  x = s^2
s = mp.findroot(lambda s: mp.tan(s) + s / 2, 20.5175)
derived["schwefel"] = {"value": 0, "at": [s ** 2], "constant": s ** 2 * mp.sin(s)}


# Eggholder, on the x1 = 512 face
def egg(a, b):
    return -(b + 47) * mp.sin(mp.sqrt(abs(a / 2 + b + 47))) - a * mp.sin(mp.sqrt(abs(a - b - 47)))


y = mp.findroot(lambda b: mp.diff(lambda t: egg(512, t), b), stated["eggholder"]["at"][1])
slope = mp.diff(lambda t: egg(t, y), 512)
derived["eggholder"] = {"value": egg(512, y), "at": [512, y]}

# Shekel, m = 10
SA = [[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
      [2, 9, 2, 9], [5, 3, 5, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, "3.6", 7, "3.6"]]
SA = [[mp.mpf(v) for v in row] for row in SA]
SC = [mp.mpf(v) / 10 for v in (1, 2, 2, 4, 4, 6, 3, 7, 5, 5)]


def shekel_q(x, i):
    return mp.fsum((x[j] - SA[i][j]) ** 2 for j in range(4)) + SC[i]


xs = newton(lambda *x: [mp.fsum(2 * (x[k] - SA[i][k]) / shekel_q(x, i) ** 2 for i in range(10))
                        for k in range(4)], stated["shekel"]["at"])
derived["shekel"] = {"value": -mp.fsum(1 / shekel_q(xs, i) for i in range(10)), "at": list(xs)}

# Hartmann 3 and 6
ALPHA = [mp.mpf(v) for v in ("1.0", "1.2", "3.0", "3.2")]
HARTMANN = {
    "hartmann3": ([[3, 10, 30], ["0.1", 10, 35], [3, 10, 30], ["0.1", 10, 35]],
                  [[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]]),
    "hartmann6": ([[10, 3, 17, "3.5", "1.7", 8], ["0.05", 10, 17, "0.1", 8, 14],
                   [3, "3.5", "1.7", 10, 17, 8], [17, 8, "0.05", 10, "0.1", 14]],
                  [[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                   [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]]),
}
for name, (Araw, Praw) in HARTMANN.items():
    A = [[mp.mpf(v) for v in row] for row in Araw]
    P = [[mp.mpf(v) / 10000 for v in row] for row in Praw]
    d = len(A[0])

    def term(x, i, A=A, P=P, d=d):
        return ALPHA[i] * mp.exp(-mp.fsum(A[i][j] * (x[j] - P[i][j]) ** 2 for j in range(d)))

    xh = newton(lambda *x, A=A, P=P, d=d, term=term:
                [mp.fsum(term(x, i) * 2 * A[i][k] * (x[k] - P[i][k]) for i in range(4)) for k in range(d)],
                stated[name]["at"])
    derived[name] = {"value": -mp.fsum(term(xh, i) for i in range(4)), "at": list(xh)}

# Branin: closed form, three minimisers
derived["branin"] = {"value": 5 / (4 * pi), "at": [-pi, mp.mpf("12.275")],
                     "all": [[-pi, mp.mpf("12.275")], [pi, mp.mpf("2.275")], [3 * pi, mp.mpf("2.475")]]}

# Michalewicz, m = 10: g_i(t) = sin t * sin(i t^2 / pi)^20, maximised per axis
M = 10


def mich_dg(i, t):
    u = i * t * t / pi
    return (mp.cos(t) * mp.sin(u) ** (2 * M)
            + mp.sin(t) * 2 * M * mp.sin(u) ** (2 * M - 1) * mp.cos(u) * 2 * i * t / pi)


grid = np.linspace(0, np.pi, 2_000_001)
h = grid[1] - grid[0]
mich_terms, mich_at = [], []
for i in range(1, 11):
    t0 = grid[int(np.argmax(np.sin(grid) * np.sin(i * grid ** 2 / np.pi) ** (2 * M)))]
    t = mp.findroot(lambda t: mich_dg(i, t), (t0 - 2 * h, t0 + 2 * h), solver="anderson")
    mich_at.append(t)
    mich_terms.append(mp.sin(t) * mp.sin(i * t * t / pi) ** (2 * M))
mich_known = {d: -mp.fsum(mich_terms[:d]) for d in range(1, 11)}

# compare ------------------------------------------------------------------
fails = []


def check(name, label, ours, truth, tol, rel=True):
    scale = max(1, abs(truth)) if rel else 1
    diff = abs(mp.mpf(ours) - truth) / scale
    ok = diff <= tol
    if not ok:
        fails.append("%s %s" % (name, label))
    print("  %-28s stated %-24r true %-26s %s" % (label, ours, mp.nstr(truth, 20),
                                                   "ok" if ok else "OFF by %s" % mp.nstr(diff, 3)))


for name in ("schwefel", "eggholder", "shekel", "hartmann3", "hartmann6", "branin"):
    print(name)
    check(name, "value", stated[name]["value"], derived[name]["value"], VALUE_TOL)
    for j, (a, b) in enumerate(zip(stated[name]["at"], derived[name]["at"])):
        check(name, "at[%d]" % j, a, b, AT_TOL, rel=False)
    for k, (pt, tr) in enumerate(zip(stated[name]["all"] or [], derived[name].get("all", []))):
        for j, (a, b) in enumerate(zip(pt, tr)):
            check(name, "all[%d][%d]" % (k, j), a, b, AT_TOL, rel=False)

print("schwefel offset constant 418.98...: true %s" % mp.nstr(derived["schwefel"]["constant"], 20))
print("eggholder df/dx1 at the optimum = %s (must be < 0)" % mp.nstr(slope, 6))
if not slope < 0:
    fails.append("eggholder slope")

print("michalewicz (m = 10)")
for d in range(1, 11):
    ours = stated["michalewicz"]["known"].get(str(d))
    if ours is None:
        print("  known[%d] missing; true %s" % (d, mp.nstr(mich_known[d], 20)))
        fails.append("michalewicz known[%d] (missing)" % d)
    else:
        check("michalewicz", "known[%d]" % d, ours, mich_known[d], VALUE_TOL)
print("  per-axis argmax: %s" % ", ".join(mp.nstr(t, 17) for t in mich_at))

if fails:
    sys.exit("\n%d stated optimum value(s) or location(s) disagree: %s" % (len(fails), ", ".join(fails)))
print("\nevery stated optimum agrees with its 50-digit derivation")
