"""Does the CNN add anything over height+weight+sex alone?

Fits a trivial linear baseline on (height, weight, gender) with NO image input,
evaluated subject-disjoint (2-fold), and compares it against the model's own
per-subject MAE from the same test split.
"""
import csv, statistics
from collections import defaultdict

ROOT = r"C:\Users\diego\Desktop\proyecto de ia3\proyecto-sastre-ia\experiments\exp_006_weight_huber"

MEAS = ["chest","waist","hip","thigh","calf","ankle","arm-length","forearm",
        "wrist","bicep","shoulder-breadth","leg-length","shoulder-to-crotch"]

# ---- load per-photo rows, aggregate to subject level ----
rows = list(csv.DictReader(open(ROOT + r"\predictions_test.csv", encoding="utf-8")))
by_subj = defaultdict(list)
for r in rows:
    by_subj[r["subject_id"]].append(r)

subjects = []
for sid, rs in by_subj.items():
    rec = {"sid": sid,
           "height": float(rs[0]["height_cm"]),
           "weight": float(rs[0]["weight_kg"]),
           "gender": float(rs[0]["gender"])}
    ok = True
    for m in MEAS:
        try:
            rec["true_" + m] = statistics.fmean(float(r["true_" + m]) for r in rs)
            rec["pred_" + m] = statistics.fmean(float(r["pred_" + m]) for r in rs)
        except (KeyError, ValueError):
            ok = False
    if ok:
        subjects.append(rec)

def solve(A, b):
    """Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            return None
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]

def fit_ols(train, target):
    feats = lambda s: [1.0, s["height"], s["weight"], s["gender"]]
    n = 4
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for s in train:
        x = feats(s); y = s["true_" + target]
        for i in range(n):
            b[i] += x[i] * y
            for j in range(n):
                A[i][j] += x[i] * x[j]
    beta = solve(A, b)
    if beta is None:
        return None
    return lambda s: sum(c * v for c, v in zip(beta, feats(s)))

# subject-disjoint 2-fold
half = len(subjects) // 2
folds = [(subjects[:half], subjects[half:]), (subjects[half:], subjects[:half])]

print(f"Sujetos en test: {len(subjects)}\n")
print(f"{'medida':<20} {'MAE modelo':>11} {'MAE baseline':>13} {'ganancia':>10}  {'veredicto'}")
print("-" * 78)

gains = []
for m in MEAS:
    model_mae = statistics.fmean(abs(s["pred_" + m] - s["true_" + m]) for s in subjects)
    errs = []
    for train, test in folds:
        f = fit_ols(train, m)
        if f is None:
            continue
        errs += [abs(f(s) - s["true_" + m]) for s in test]
    base_mae = statistics.fmean(errs)
    gain = base_mae - model_mae
    gains.append((m, model_mae, base_mae, gain))
    pct = gain / base_mae * 100 if base_mae else 0
    verdict = "aporta" if pct > 10 else ("marginal" if pct > 3 else "NO APORTA")
    print(f"{m:<20} {model_mae:>9.2f}cm {base_mae:>11.2f}cm {gain:>+8.2f}cm  {verdict} ({pct:+.0f}%)")

print()
agg_model = statistics.fmean(g[1] for g in gains)
agg_base = statistics.fmean(g[2] for g in gains)
print(f"MAE agregado modelo  : {agg_model:.3f} cm")
print(f"MAE agregado baseline: {agg_base:.3f} cm   (sin usar NI UNA foto)")

# ---- ¿cuanto gana una correccion de sesgo post-hoc (2-fold, honesta)? ----
print("\n--- Correccion de sesgo post-hoc (sesgo estimado en mitad A, aplicado en mitad B) ---")
print(f"{'medida':<20} {'MAE actual':>11} {'MAE -sesgo':>11} {'gana':>8}")
print("-" * 55)
tot_before = tot_after = 0.0
for m in MEAS:
    before, after = [], []
    for train, test in folds:
        bias = statistics.fmean(s["pred_" + m] - s["true_" + m] for s in train)
        for s in test:
            before.append(abs(s["pred_" + m] - s["true_" + m]))
            after.append(abs((s["pred_" + m] - bias) - s["true_" + m]))
    b, a = statistics.fmean(before), statistics.fmean(after)
    tot_before += b; tot_after += a
    flag = "  <--" if b - a > 0.15 else ""
    print(f"{m:<20} {b:>9.2f}cm {a:>9.2f}cm {b-a:>+7.2f}{flag}")
print(f"\nAgregado: {tot_before/len(MEAS):.3f} cm  ->  {tot_after/len(MEAS):.3f} cm")
