"""TFPS-1 variance-triggered extension (PREREG r3.1 §7). Reads ONLY per-seed primary scores; judges widths only.

T(H) = OR_{a in {gauss, unit}, both non-BLOCKED} [ t_{.95,nu} * se(a, zero) >= m_H ]
    OR OR_{non-BLOCKED a in {gauss, unit, zero}, gating B} [ (t_{.95,nu} + t_{1-0.05/k,nu}) * se(a, B) >= m_{H,B} ]
se, nu: Welch-Satterthwaite from released-selection TFPS scores and lr-searched baseline scores, with the seeds each
arm has. se == 0 -> width 0. BLOCKED arms and non-gating pairs excluded.

Usage: python extension_trigger.py scores.json  (prints JSON; never prints a mean or a difference)
scores.json: {"H": 192, "m_H": 0.011, "m_HB": {"patchtst": 0.037, "dlinear": 0.011},
              "scores": {"tfps_gauss": [...], "tfps_gauss_unit": [...], "tfps_zero": [...], "patchtst": [...], "dlinear": [...]},
              "blocked": ["..."]}
"""
import json, math, sys
from scipy import stats

TFPS_Q2 = ("tfps_gauss", "tfps_gauss_unit")
TFPS_ALL = ("tfps_gauss", "tfps_gauss_unit", "tfps_zero")


def _var(x):
    n = len(x); m = sum(x) / n
    return sum((v - m) ** 2 for v in x) / (n - 1)


def welch_se_nu(x, y):
    a, b = _var(x) / len(x), _var(y) / len(y)
    se2 = a + b
    if se2 == 0:
        return 0.0, None
    nu = se2 ** 2 / ((a ** 2) / (len(x) - 1) + (b ** 2) / (len(y) - 1))
    return math.sqrt(se2), nu


def trigger(spec):
    sc, blocked = spec["scores"], set(spec.get("blocked", []))
    gating = [B for B, m in spec["m_HB"].items() if m > 0]
    k = len(gating)
    q_u = 1 - 0.05 / k if k else None
    terms = []
    for a in TFPS_Q2:
        if a in blocked or "tfps_zero" in blocked:
            continue
        se, nu = welch_se_nu(sc[a], sc["tfps_zero"])
        w = 0.0 if se == 0 else stats.t.ppf(0.95, nu) * se
        terms.append({"pair": f"{a}-tfps_zero", "width": w, "margin": spec["m_H"], "fires": w >= spec["m_H"]})
    for a in TFPS_ALL:
        if a in blocked:
            continue
        for B in gating:
            if B in blocked:
                continue
            se, nu = welch_se_nu(sc[a], sc[B])
            w = 0.0 if se == 0 else (stats.t.ppf(0.95, nu) + stats.t.ppf(q_u, nu)) * se
            m = spec["m_HB"][B]
            terms.append({"pair": f"{a}-{B}", "width": w, "margin": m, "fires": w >= m})
    return {"H": spec["H"], "T": any(t["fires"] for t in terms), "k_gating": k, "terms": terms}


def _selftest():
    import random
    r = random.Random(0)
    base = {"H": 192, "m_H": 0.011, "m_HB": {"patchtst": 0.037, "dlinear": 0.011}, "blocked": []}
    # 1. tiny sds -> no extension
    s = {a: [0.42 + r.gauss(0, 1e-4) for _ in range(5)] for a in TFPS_ALL}
    s.update(patchtst=[0.46 + r.gauss(0, 1e-4) for _ in range(5)], dlinear=[0.434 + r.gauss(0, 1e-5) for _ in range(5)])
    assert not trigger(dict(base, scores=s))["T"]
    # 2. large TFPS sd -> extension
    s2 = dict(s); s2["tfps_zero"] = [0.42 + r.gauss(0, 0.02) for _ in range(5)]
    out = trigger(dict(base, scores=s2)); assert out["T"]
    # 3. hand-checked width: x sd 0.01 n5, y sd 0 n5 -> se = 0.01/sqrt5, nu = 4
    x = [0.0, 0.01, 0.02, 0.03, 0.04]; sd = math.sqrt(_var(x)); y = [0.5] * 5
    se, nu = welch_se_nu(x, y)
    assert abs(se - sd / math.sqrt(5)) < 1e-12 and abs(nu - 4) < 1e-9
    # 4. se == 0 -> width 0, no crash
    z = {a: [0.4] * 5 for a in TFPS_ALL}; z.update(patchtst=[0.45] * 5, dlinear=[0.43] * 5)
    o = trigger(dict(base, scores=z)); assert not o["T"] and all(t["width"] == 0 for t in o["terms"])
    # 5. non-gating baseline excluded, k adapts
    o = trigger(dict(base, m_HB={"patchtst": 0.037, "dlinear": -0.002}, scores=s))
    assert o["k_gating"] == 1 and not any("dlinear" in t["pair"] for t in o["terms"])
    # 6. blocked arm excluded; blocked zero removes both Q2 terms
    o = trigger(dict(base, blocked=["tfps_zero"], scores=s2))
    assert not any("tfps_zero" in t["pair"] for t in o["terms"])
    # 7. n = 4 arm works
    s4 = dict(s); s4["tfps_gauss"] = s["tfps_gauss"][:4]; trigger(dict(base, scores=s4))
    return "extension_trigger selftest: 7/7 PASS"


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        print(_selftest())
    else:
        print(json.dumps(trigger(json.load(open(sys.argv[1]))), indent=1))
