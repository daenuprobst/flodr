"""The hidden-contrast field h(y) = I(G ; R | Y=y), checked against a permutation null:
G shuffled WITHIN screen bins destroys the contrast while preserving its screen marginal.

Run: [DATASETS=bmarrow,cerebellum,plant] .venv/bin/python scripts/ablations/hidden_contrast.py
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, SCRIPTS)

from flodr import FloDR                                          # noqa: E402

SCRNA = os.path.join(SCRIPTS, "cache", "scrna")
OUT = os.path.join(SCRIPTS, "cache", "hidden_contrast.json")
SEED, W = 0, 2.0


def brief(cert):
    keep = ("passed", "certifies", "slope", "r2", "level", "dynamic_range", "n_bins",
            "p_level", "n_perm", "shape_ok",
            # the 99 permutation values, kept so the null mean -- and hence the uncorrected
            # gap, which is mean_gap + null mean -- can be recovered without a re-fit
            "null_level",
            "confidence", "mean_gap")
    return {k: (round(float(cert[k]), 4) if isinstance(cert[k], (int, float))
                and not isinstance(cert[k], bool) else cert[k]) for k in keep if k in cert}


def brief_sigma(cert):
    """sigma's verdict fields. Its level lives in ratio_quartiles, not `level`."""
    quart = cert.get("ratio_quartiles", [None, None, None])
    return {"passed": cert.get("passed"), "certifies": cert.get("certifies"),
            "level": None if quart[1] is None else round(float(quart[1]), 3),
            "slope": round(float(cert.get("slope", float("nan"))), 3),
            "confidence": cert.get("confidence"),
            "dynamic_range": round(float(cert.get("dynamic_range", float("nan"))), 2)}


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for dset in os.environ.get("DATASETS", "bmarrow,cerebellum,plant").split(","):
        if os.path.exists(os.path.join(SCRNA, f"{dset}.npz")):
            npz = np.load(os.path.join(SCRNA, f"{dset}.npz"), allow_pickle=True)
            pts, labels = npz["P"].astype(np.float32), npz["labels"].astype(str)
        else:                       # benchmark-scale sets, for the null-spread vs n check
            from datasets import ALL
            X_raw, labels, _t, _ = ALL[dset]()
            pts, labels = np.asarray(X_raw, np.float32), np.asarray(labels).astype(str)
        # drop types too rare to classify: the estimand is undefined where a class has no
        # held-out support, and keeping them turns the certificate into a study of rare labels
        kinds, counts = np.unique(labels, return_counts=True)
        keep = np.isin(labels, kinds[counts >= 200])
        pts, labels = pts[keep], labels[keep]
        print(f"\n=== {dset}: {len(pts):,} cells, {len(np.unique(labels))} types ===", flush=True)

        t0 = time.perf_counter()
        torch._dynamo.reset()
        model = FloDR(w=W, random_state=SEED, device="cuda", density=True,
                      advanced=dict(gate_max=0.5)).fit(pts)
        print(f"  fit {time.perf_counter()-t0:.0f}s", flush=True)

        # one call for every display and certificate, off the SAME fit: raw_recipe trades
        # bitwise reproducibility for speed on cuda, so a second fit at the same seed can move
        # the layout and fields taken from different fits are not the same map
        t0 = time.perf_counter()
        diag = model.diagnostics(G=labels)
        sigma, sigma_cert = diag["spread"]["field"].astype(np.float32), diag["spread"]["cert"]
        field, cert = diag["hidden_contrast"]["field"], diag["hidden_contrast"]["cert"]
        cert["field_null"] = diag["hidden_contrast"]["field_null"]
        print(f"  sigma  median {np.median(sigma):.2f}  passed={sigma_cert.get('passed')} "
              f"level={sigma_cert.get('ratio_quartiles', [None, None, None])[1]} "
              f"conf={sigma_cert.get('confidence')}", flush=True)
        _, kept_cnt = np.unique(labels, return_counts=True)
        freq = kept_cnt / kept_cnt.sum()
        entry = dict(brief(cert), field_mean=float(np.mean(field)),
                     field_median=float(np.median(field)),
                     field_p90=float(np.percentile(field, 90)),
                     # entropy of the kept labels, the scale the field is read against: h(y) is
                     # reported in the paper as a share of what the annotation says about type
                     label_entropy=float(-(freq * np.log(freq)).sum()), n_types=int(len(freq)),
                     secs=time.perf_counter() - t0, n_cells=int(len(pts)),
                     null_sd=float(np.std(np.asarray(cert["null_level"], float), ddof=1)))
        fields = {"real": field.astype(np.float32),
                  "permuted-null": np.asarray(cert["field_null"], np.float32)}
        print(f"  debiased signal   mean {cert['mean_gap']:+.4f} nats", flush=True)
        nulls = np.asarray(cert["null_level"], float)
        print(f"  permutation null  n={len(nulls)}  sd {nulls.std(ddof=1):.5f}  "
              f"range [{nulls.min():+.4f}, {nulls.max():+.4f}]  "
              f"signal/nullsd {cert['mean_gap']/max(nulls.std(ddof=1),1e-12):.1f}x", flush=True)
        print(f"  level  p={cert['p_level']:.3f} (n_perm={cert['n_perm']})   "
              f"shape slope={cert['slope']:.3f} r2={cert['r2']:.3f} "
              f"ok={cert['shape_ok']}", flush=True)
        print(f"  -> passed={cert['passed']}  confidence={cert['confidence']:.3f}", flush=True)
        # both fields on one file so the null can be drawn on the real field's colour scale,
        # which is the only way a reader can see that the control actually collapsed
        np.savez_compressed(
            os.path.join(SCRIPTS, "cache", f"hidden_contrast_{dset}.npz"),
            field=fields["real"], field_null=fields["permuted-null"],
            sigma=sigma, sigma_cert=np.array(json.dumps(brief_sigma(sigma_cert), default=float)),
            Y=np.asarray(model.embedding_, np.float32), labels=labels,
            cert=np.array(json.dumps(entry, default=float)))
        res[dset] = entry
        json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n-> {OUT}", flush=True)
    print("the null must collapse the field and refuse; if it passes, the field is reading "
          "label imbalance rather than hidden information")


if __name__ == "__main__":
    main()
