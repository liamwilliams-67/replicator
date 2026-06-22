#!/usr/bin/env bash
# Replicator M3 serve-stack spike (go/no-go).
#
# RUN THIS ON THE CPU SERVE BOX. It:
#   1. builds (or locates) llama.cpp,
#   2. downloads the stock base models, converts them to GGUF and quantizes to
#      Q4_K_M + Q8_0 (the quants we'd actually ship),
#   3. benchmarks prefill/generation t/s per model per quant,
#   4. measures whether llama.cpp REUSES the prompt cache across turns or
#      re-prefills the whole prompt every turn (the Gated-DeltaNet blocker),
#   5. prints a go/no-go verdict + a context-budget recommendation.
#
# Idempotent: re-runs skip any step whose output already exists.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$HERE/config.env"

log(){ printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die(){ printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
have(){ command -v "$1" >/dev/null 2>&1; }

mkdir -p "$WORK" "$RESULTS"
have python3 || die "python3 not found"
have git || die "git not found"

# --- 1. llama.cpp -----------------------------------------------------------
if   [[ -n "${LLAMA_CPP_DIR:-}" && -x "$LLAMA_CPP_DIR/llama-bench" ]]; then
  BIN="$LLAMA_CPP_DIR"; LLAMA_SRC="${LLAMA_CPP_SRC:-$LLAMA_CPP_DIR}"
elif [[ -n "${LLAMA_CPP_DIR:-}" && -x "$LLAMA_CPP_DIR/build/bin/llama-bench" ]]; then
  BIN="$LLAMA_CPP_DIR/build/bin"; LLAMA_SRC="$LLAMA_CPP_DIR"
else
  LLAMA_SRC="$WORK/llama.cpp"
  if [[ ! -d "$LLAMA_SRC/.git" ]]; then
    log "Cloning llama.cpp"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_SRC"
  fi
  if [[ ! -x "$LLAMA_SRC/build/bin/llama-bench" ]]; then
    have cmake || die "cmake not found (needed to build llama.cpp)"
    log "Building llama.cpp (CPU, native)"
    cmake -S "$LLAMA_SRC" -B "$LLAMA_SRC/build" \
      -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
    cmake --build "$LLAMA_SRC/build" -j --config Release
  fi
  BIN="$LLAMA_SRC/build/bin"
fi

BENCH="$BIN/llama-bench"
SERVER="$BIN/llama-server"
QUANTIZE="$BIN/llama-quantize"
CONVERT="$LLAMA_SRC/convert_hf_to_gguf.py"
[[ -x "$BENCH" ]]    || die "llama-bench not found at $BENCH"
[[ -x "$SERVER" ]]   || die "llama-server not found at $SERVER"
[[ -x "$QUANTIZE" ]] || die "llama-quantize not found at $QUANTIZE"
[[ -f "$CONVERT" ]]  || die "convert_hf_to_gguf.py not found at $CONVERT"
log "Using llama.cpp binaries in $BIN"

# --- deps -------------------------------------------------------------------
log "Installing python deps"
python3 -m pip install -q -r "$HERE/requirements.txt"
REQ="$LLAMA_SRC/requirements/requirements-convert_hf_to_gguf.txt"
[[ -f "$REQ" ]] && python3 -m pip install -q -r "$REQ" || true

# --- 2. download -> convert -> quantize -------------------------------------
for entry in "${MODELS[@]}"; do
  label="${entry%%|*}"; hfid="${entry##*|}"
  hfdir="$WORK/$label-hf"; f16="$WORK/$label-f16.gguf"
  if [[ ! -f "$f16" ]]; then
    if [[ ! -d "$hfdir" ]]; then
      log "Download $hfid"
      huggingface-cli download "$hfid" --local-dir "$hfdir" \
        --exclude "*.pth" "original/*" "consolidated*"
    fi
    log "Convert $label -> f16 GGUF"
    python3 "$CONVERT" "$hfdir" --outfile "$f16" --outtype f16
  fi
  for q in "${QUANTS[@]}"; do
    out="$WORK/$label-$q.gguf"
    [[ -f "$out" ]] || { log "Quantize $label -> $q"; "$QUANTIZE" "$f16" "$out" "$q"; }
  done
done

# --- 3. benchmark (per model per quant) -------------------------------------
for entry in "${MODELS[@]}"; do
  label="${entry%%|*}"
  for q in "${QUANTS[@]}"; do
    gguf="$WORK/$label-$q.gguf"
    log "Bench $label $q"
    python3 "$HERE/spike.py" bench --bench "$BENCH" --model "$gguf" \
      --label "$label" --quant "$q" --threads "$THREADS" \
      --ctx "$CTX_LIST" --ngen "$NGEN" --out "$RESULTS"
  done
done

# --- 4. cache-reuse test (per model; architectural, so Q4 is enough) --------
for entry in "${MODELS[@]}"; do
  label="${entry%%|*}"
  gguf="$WORK/$label-Q4_K_M.gguf"
  log "Cache-reuse test $label"
  python3 "$HERE/spike.py" cachetest --server "$SERVER" --model "$gguf" \
    --label "$label" --threads "$THREADS" --ctx "$SERVER_CTX" \
    --base-tokens "$BASE_TOKENS" --port "$PORT" --out "$RESULTS"
done

# --- 5. report --------------------------------------------------------------
log "Report"
python3 "$HERE/spike.py" report --results "$RESULTS" --budget "$BUDGET_S"
log "Done — full report at $RESULTS/REPORT.md"
