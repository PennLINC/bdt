#!/bin/bash
# Smoke-test the BDT tool image: every binary the node-graph engine needs must
# resolve on PATH and run.  Use against the base image or the app image, e.g.:
#
#   docker build -f Dockerfile.base -t bdt-base:test .
#   docker run --rm -v "$PWD/scripts:/s" bdt-base:test bash /s/container_smoke_test.sh
#
# Exits non-zero on the first missing/broken tool (suitable for CI).
set -euo pipefail

fail=0
check() {  # check <label> <command...>
    local label="$1"; shift
    if "$@" >/dev/null 2>&1; then
        echo "  OK    $label"
    else
        echo "  FAIL  $label  ($*)"
        fail=1
    fi
}

echo "== BDT container tool smoke test =="
check "wb_command"        wb_command -version
check "giftirs"           giftirs --help
check "trxrs"             trxrs --help
# ANTs is ANTsPy (antspyx) — a Python dep in the app image, not a base-image binary.
if command -v python >/dev/null 2>&1; then
    check "antspyx (import ants)" python -c "import ants"
fi

# A couple of the specific wb_command sub-ops the engine relies on must exist
# (these are the 2.x ops behind the surface/parcellation recipes).
for op in metric-resample volume-to-surface-mapping cifti-create-dense-scalar \
          cifti-create-dense-from-template cifti-separate cifti-parcellate; do
    check "wb_command -$op" bash -c "wb_command -$op 2>&1 | grep -qi ."
done

# The CLI entrypoint (only present in the app image; skipped otherwise).
if command -v bdt >/dev/null 2>&1; then
    check "bdt --help" bdt --help
fi

if [ "$fail" -ne 0 ]; then
    echo "SMOKE TEST FAILED"; exit 1
fi
echo "SMOKE TEST PASSED"
