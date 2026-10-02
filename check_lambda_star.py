"""Optional: how far lambda* is from the first study's start (0.5) on the old headline
instances, in clipped subgradient steps of 0.02.  Usage: python3 check_lambda_star.py [instance dir]"""
import glob, gzip, json, os, statistics as st, sys
d = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/mstkp_final/instances/n300_d0.050000_b0.150_k+0.0000_core")
lam = [json.load(gzip.open(p, "rt"))["meta"]["plain_lr_lambda"] for p in sorted(glob.glob(d + "/*.json.gz"))]
steps = [max(0.0, (l - 0.5) / 0.02) for l in lam]
print(f"{len(lam)} instances: lambda* median {st.median(lam):.3f} (min {min(lam):.3f}, max {max(lam):.3f})")
print(f"clipped steps (0.02) needed from lambda0 = 0.5: median {st.median(steps):.0f}, min {min(steps):.0f}")
