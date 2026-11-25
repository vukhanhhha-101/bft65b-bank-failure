import json, re, sys
from pathlib import Path

nb_path = Path(sys.argv[1]) if len(sys.argv) > 1 else next(Path(".").glob("*.ipynb"))
root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
nb = json.loads(nb_path.read_text(encoding="utf-8")); code = [c for c in nb["cells"] if c["cell_type"] == "code"]
src_all = "\n".join("".join(c["source"]) for c in code); problems = []

scripts = sorted(set(re.findall(r"src/([\w\-]+\.py)", src_all)) | set(re.findall(r"[\[\"'](?:\.?/)?src/([\w\-]+\.py)", src_all)))
print(f"[1] scripts referenced by the notebook ({len(scripts)}):", ", ".join(scripts))
missing = [s for s in scripts if not (root / "src" / s).exists()]
if missing: problems.append(f"MISSING in repo/src: {missing}")

errs = [(i, o.get("ename"), o.get("evalue", "")[:120]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
print(f"[2] cells with a saved error output: {errs if errs else 'none'}")
if errs: problems.append(f"saved errors: {errs}")

installs = dict(re.findall(r"([A-Za-z0-9_\-]+)==([0-9][\w\.]*)", " ".join(l for l in src_all.splitlines() if "pip install" in l)))
req = root / "requirements.txt"
if not req.exists(): problems.append("requirements.txt missing")
else:
    pins = {k.lower(): v for k, v in re.findall(r"^([A-Za-z0-9_\-]+)==([\w\.]+)", req.read_text(), re.M)}
    for k, v in installs.items():
        if pins.get(k.lower()) != v: problems.append(f"VERSION DRIFT {k}: notebook installs {v}, requirements.txt pins {pins.get(k.lower())}")
    print(f"[3] notebook pip pins {installs} vs requirements.txt -> {'OK' if not [p for p in problems if 'DRIFT' in p] else 'MISMATCH'}")
    if "-r requirements.txt" not in src_all: problems.append("notebook does not install from requirements.txt (use: !pip install -q -r requirements.txt)")

if "drive.mount" in src_all and "git clone" not in src_all:
    problems.append("notebook only finds the repo through Google Drive; a grader cannot. Add a fallback: git clone <public GitHub URL> when the Drive folder is absent")
print("[4] Drive-only repo location:", "YES (problem)" if "drive.mount" in src_all and "git clone" not in src_all else "no")

readme = root / "README.md"
if readme.exists():
    t = readme.read_text(encoding="utf-8").lower()
    for need in ("requirements.txt", "claude", "ai"):
        if need not in t: problems.append(f"README does not mention '{need}' (AI-tool citation / install step)")
else: problems.append("README.md missing")
gi = root / ".gitignore"
if not gi.exists() or "data_raw" not in gi.read_text(): problems.append(".gitignore does not exclude data_raw/ (data files must not be committed)")

print("\n" + ("ALL CHECKS PASSED" if not problems else "PROBLEMS:\n - " + "\n - ".join(problems)))
sys.exit(1 if problems else 0)
