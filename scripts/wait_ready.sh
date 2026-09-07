#!/usr/bin/env bash
# Poll OpenAI-compat /v1/models. First boot can take an hour (PLE fill).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

HOST="${HOST:-127.0.0.1}"
WAIT="${WAIT:-3600}"
INTERVAL="${INTERVAL:-15}"
BASE="http://${HOST}:${PORT}"

echo "waiting for ${BASE}/v1/models up to ${WAIT}s (interval ${INTERVAL}s)"
deadline=$((SECONDS + WAIT))
while (( SECONDS < deadline )); do
  if curl -sf -m 5 "${BASE}/v1/models" >/dev/null 2>&1; then
    echo "ready after ${SECONDS}s"
    exit 0
  fi
  echo "not ready (${SECONDS}s) container=$(docker inspect -f '{{.State.Status}}' "${CONTAINER}" 2>/dev/null || echo missing)"
  sleep "${INTERVAL}"
done
echo "not ready after ${WAIT}s" >&2
exit 1
