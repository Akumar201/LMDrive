#!/usr/bin/env bash
# fix_env.sh — One-shot environment repair for LMDrive
#
# Run this from the LMDrive project root with the lmdrive conda env active:
#   conda activate lmdrive
#   bash fix_env.sh
#
# Safe to re-run; skips steps that are already correct.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PIP="$(which pip)"
PYTHON="$(which python)"

# pip_version <package> — reads installed version via pip show (never imports the package)
# Returns empty string if not installed. The || true guards against pipefail on missing packages.
pip_version() {
    $PIP show "$1" 2>/dev/null | awk '/^Version:/ { print $2 }' || true
}

version_gte() {
    # returns 0 (true) if $1 >= $2
    $PYTHON -c "from packaging.version import Version; exit(0 if Version('$1') >= Version('$2') else 1)" 2>/dev/null
}

echo ""
echo "=== LMDrive environment fix ==="
echo "Python : $PYTHON  ($($PYTHON --version 2>&1))"
echo "pip    : $PIP"
echo ""

# ---------------------------------------------------------------------------
# 1. vision_encoder / timm  (custom fork — must be installed from source)
# ---------------------------------------------------------------------------
if ! $PYTHON -c "import timm" 2>/dev/null; then
    echo "[1/4] Installing vision_encoder (timm custom fork)..."
    cd vision_encoder && $PIP install -e . --quiet && cd "$SCRIPT_DIR"
    echo "      Done."
else
    echo "[1/4] timm already importable — skipping."
fi

# ---------------------------------------------------------------------------
# 2. peft — must be <0.11.0 for transformers<4.39.0
#    Use pip show (not import) because a broken peft crashes on import.
# ---------------------------------------------------------------------------
PEFT_VERSION="$(pip_version peft)"
TRANSFORMERS_VERSION="$(pip_version transformers)"

if [ -z "$PEFT_VERSION" ]; then
    echo "[2/4] peft not installed — installing peft==0.10.0..."
    $PIP install "peft==0.10.0" --quiet
    echo "      Done."
elif version_gte "$PEFT_VERSION" "0.11.0" && ! version_gte "$TRANSFORMERS_VERSION" "4.39.0"; then
    echo "[2/4] peft $PEFT_VERSION is too new for transformers $TRANSFORMERS_VERSION — downgrading to peft==0.10.0..."
    $PIP install "peft==0.10.0" --quiet
    echo "      Done."
else
    echo "[2/4] peft $PEFT_VERSION OK with transformers $TRANSFORMERS_VERSION — skipping."
fi

# ---------------------------------------------------------------------------
# 3. LAVIS (install after peft is fixed, since lavis.__init__ imports peft)
# ---------------------------------------------------------------------------
if ! $PYTHON -c "from lavis.common.registry import registry" 2>/dev/null; then
    echo "[3/4] Installing LAVIS..."
    cd LAVIS && $PIP install -e . --quiet && cd "$SCRIPT_DIR"
    echo "      Done."
else
    echo "[3/4] LAVIS already importable — skipping."
fi

# ---------------------------------------------------------------------------
# 4. leaderboard + scenario_runner runtime deps
#    leaderboard/requirements.txt is the canonical list (dictor, easydict, pygame,
#    py_trees, Shapely, networkx>=3.0, tabulate, xmlschema, ephem, six, …).
#    Skip: opencv-python (we use headless 4.5.5.64) and bitsandbytes (pinned below).
# ---------------------------------------------------------------------------
if ! $PYTHON -c "import dictor, py_trees, easydict" 2>/dev/null; then
    echo "[4/5] Installing leaderboard runtime dependencies..."
    grep -v "^opencv-python" leaderboard/requirements.txt | \
        grep -v "^bitsandbytes" | \
        $PIP install -r /dev/stdin --quiet
    echo "      Done."
else
    echo "[4/5] leaderboard deps already installed — skipping."
fi

# ---------------------------------------------------------------------------
# 5. bitsandbytes — version depends on system CUDA
#    CUDA 11.x → 0.41.3  (uses libcusparse.so.11)
#    CUDA 12.x → 0.43.3  (uses libcusparse.so.12; 0.41.x fails with "No such file")
# ---------------------------------------------------------------------------
SYSTEM_CUDA_MAJOR=""
if command -v nvcc >/dev/null 2>&1; then
    SYSTEM_CUDA_MAJOR=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1 || true)
fi

if [ "$SYSTEM_CUDA_MAJOR" = "12" ]; then
    BNB_TARGET="0.43.3"
    BNB_MIN="0.43.0"
else
    BNB_TARGET="0.41.3"
    BNB_MIN="0.41.0"
fi

BNB_VERSION="$(pip_version bitsandbytes)"

_needs_bnb=false
if [ -z "$BNB_VERSION" ]; then
    _needs_bnb=true
elif ! version_gte "$BNB_VERSION" "$BNB_MIN"; then
    _needs_bnb=true
fi

if $_needs_bnb; then
    echo "[5/5] Installing bitsandbytes==${BNB_TARGET} (CUDA ${SYSTEM_CUDA_MAJOR:-11}.x)..."
    $PIP install "bitsandbytes==${BNB_TARGET}" --quiet
    echo "      Done."
else
    echo "[5/5] bitsandbytes ${BNB_VERSION} OK (CUDA ${SYSTEM_CUDA_MAJOR:-11}.x) — skipping."
fi

# ---------------------------------------------------------------------------
# Final check
# ---------------------------------------------------------------------------
echo ""
echo "=== Verification ==="
$PYTHON - <<'EOF'
import sys

failures = []

try:
    import timm
    print(f"  timm              OK  ({timm.__version__})")
except Exception as e:
    failures.append(f"timm: {e}")
    print(f"  timm              FAIL: {e}")

try:
    from lavis.common.registry import registry
    print(f"  lavis.registry    OK")
except Exception as e:
    failures.append(f"lavis: {e}")
    print(f"  lavis.registry    FAIL: {e}")

try:
    import peft
    print(f"  peft              OK  ({peft.__version__})")
except Exception as e:
    failures.append(f"peft: {e}")
    print(f"  peft              FAIL: {e}")

try:
    import bitsandbytes as bnb
    ver = getattr(bnb, '__version__', None) or getattr(bnb, 'version', None) or 'installed'
    print(f"  bitsandbytes      OK  ({ver})")
except Exception as e:
    failures.append(f"bitsandbytes: {e}")
    print(f"  bitsandbytes      FAIL: {e}")

try:
    import transformers
    print(f"  transformers      OK  ({transformers.__version__})")
except Exception as e:
    failures.append(f"transformers: {e}")
    print(f"  transformers      FAIL: {e}")

print()
if failures:
    print("Some checks FAILED — see above.")
    sys.exit(1)
else:
    print("All checks passed. Run:")
    print("  python3 leaderboard/team_code/lmdrive_inference_server.py")
EOF
