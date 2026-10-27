"""Summarise gtest EXPECT_NEAR failures from NVIDIA's FK unit test (per field, sign flips, worst diff)."""
import collections, re, sys
L = open(sys.argv[1]).read().splitlines()
c, flips, worst, vals = collections.Counter(), collections.Counter(), collections.defaultdict(float), []
num = lambda s: float(re.search(r"evaluates to (\S+?),? ?(and)?$", s).group(1).rstrip(","))
for i, l in enumerate(L):
    if l.startswith("The difference between seq."):
        name = l.split("seq.")[1].split("(")[0]; d = float(l.split(" is ")[1].split(",")[0])
        a, b = num(L[i + 1]), num(L[i + 2]); c[name] += 1; worst[name] = max(worst[name], d)
        if abs(a + b) < 1e-4: flips[name] += 1
        vals.append((name, round(a, 6), round(b, 6)))
print("failures:", dict(c), "| sign flips:", dict(flips), "| worst:", {k: round(v, 4) for k, v in worst.items()})
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write("\n".join(f"{n} {a} {b}" for n, a, b in vals))
