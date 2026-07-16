#!/usr/bin/env bash
# Phase 2 / F1 -- batched MMseqs2 search of the deduped chain work-list against a
# taxonomy-bearing UniRef db, producing one a3m per chain keyed by chain_key.
#
# This is the EXACT pipeline that produced the Step-0 test-set a3m (median paired
# depth 928), lifted into a script so the full le600 run is reproducible.
#
# RUN INSIDE WSL (ext4), never from Windows -- mmseqs random IO on /mnt/c is slow.
# WSL outbound network is dead on this host; nothing here fetches anything.
#
#   bash scripts/data/run_msa_search.sh \
#       /mnt/c/Users/costa/src/tinyfold/data/processed/msa/chains_le600.fasta \
#       $HOME/msa_db/uniref50_db \
#       $HOME/msa_db/le600 \
#       $HOME/msa_db/le600/out
#
# Timing anchor: 110 chains ~34 min disk-backed => budget hours for 4,750. Run bg.
set -euo pipefail

QUERY_FASTA="${1:?query fasta (a chain-key FASTA; may live on /mnt/c)}"
DB="${2:-$HOME/msa_db/uniref50_db}"
WORK="${3:-$HOME/msa_db/le600}"          # ext4 scratch (createdb + search tmp)
OUT="${4:-$WORK/out}"                    # <chain_key>.a3m land here
THREADS="$(nproc 2>/dev/null || echo 8)"

export PATH="$HOME/mmseqs/bin:$PATH"
command -v mmseqs >/dev/null 2>&1 || { echo "ERROR: mmseqs not on PATH"; exit 1; }
[ -s "${DB}" ] && [ -s "${DB}_h" ] || { echo "ERROR: db or header db empty: ${DB}"; exit 1; }

mkdir -p "$WORK" "$OUT"
# Copy the query onto ext4 (createdb is fine off /mnt/c, but keep everything local).
LOCAL_FASTA="$WORK/query.fasta"
cp -f "$QUERY_FASTA" "$LOCAL_FASTA"
NQ=$(grep -c '^>' "$LOCAL_FASTA")
echo "[F1] $(date -u +%H:%M:%S) query chains: $NQ | db: $DB | threads: $THREADS"

QDB="$WORK/qdb"
RES="$WORK/res"
MSA="$WORK/msa"
TMP="$WORK/tmp"
rm -rf "$TMP"; mkdir -p "$TMP"

# 1) query db
mmseqs createdb "$LOCAL_FASTA" "$QDB" >/dev/null
# 2) profile-ish search: --db-load-mode 0 keeps the 45 GB db disk-backed (32 GB
#    RAM); 2 iterations + max-seqs 3000 mirror the Step-0 run exactly.
echo "[F1] $(date -u +%H:%M:%S) mmseqs search ..."
mmseqs search "$QDB" "$DB" "$RES" "$TMP" \
    -s 5.7 --db-load-mode 0 --num-iterations 2 --max-seqs 3000 \
    --threads "$THREADS"
# 3) MSA with taxonomy (msa-format-mode 2 = a3m with headers => TaxID= survives)
echo "[F1] $(date -u +%H:%M:%S) result2msa ..."
mmseqs result2msa "$QDB" "$DB" "$RES" "$MSA" --msa-format-mode 2 --threads "$THREADS"
# 4) unpack to <chain_key>.a3m (name-mode 1 = use the lookup name = the FASTA id)
echo "[F1] $(date -u +%H:%M:%S) unpackdb ..."
mmseqs unpackdb "$MSA" "$OUT" --unpack-name-mode 1 --unpack-suffix .a3m

NA=$(ls "$OUT"/*.a3m 2>/dev/null | wc -l)
echo "[F1] $(date -u +%H:%M:%S) DONE: $NA a3m in $OUT (queried $NQ)"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$WORK/DONE"
