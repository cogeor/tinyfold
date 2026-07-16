#!/usr/bin/env bash
# MMseqs2 + taxonomy-bearing sequence DB for cross-chain coevolution (Phase 2, M1).
#
# RUN THIS ONLY AFTER STEP 0 PASSES:
#     uv run python scripts/data/measure_msa_depth.py --help
# Step 0 measures paired depth against the ColabFold PUBLIC server for ~200 test
# complexes (fair use, zero download). If paired depth is thin, DO NOT run this.
#
# ---------------------------------------------------------------------------
# THE CONSTRAINTS (measured on this machine, 2026-07-16)
# ---------------------------------------------------------------------------
#   Disk free : 245 GB      <-- binding
#   RAM       : 31.7 GB     <-- binding (MMseqs2 wants the DB resident)
#   CPU       : 24 cores / 32 threads
#
# Verified download sizes (HTTP content-length):
#   uniref30_2302.db.tar.gz        99.4 GB   taxonomy: YES   RISKY: extracted
#                                                            size unverified;
#                                                            ColabFold states
#                                                            ~1 TB for
#                                                            uniref30+envdb.
#   colabfold_envdb_202108.db.tar  119.7 GB  taxonomy: NO  <-- USELESS HERE.
#   uniref90.fasta.gz               29.9 GB  taxonomy: YES (TaxID=)  ~150 GB db
#   uniref50.fasta.gz                8.2 GB  taxonomy: YES (TaxID=)  ~45 GB db  <- VERIFIED
#
# WHY envdb IS EXCLUDED ON THE MERITS, NOT JUST SIZE: pairing is a JOIN ON
# TAXONOMY. BFD/metagenomic sequences have no reliable species assignment, so
# they can deepen a per-chain profile but contribute ZERO cross-chain signal.
# This is the same reason DIPS-Plus's free 11.17 GB HHblits-vs-BFD MSA tarball
# does not help us.
#
# DEFAULT = uniref50: it fits comfortably and is the honest first cut. Move up
# the ladder (uniref90 -> uniref30) only if Step 0 says depth is the limiter.
#
# Usage:  bash scripts/data/setup_msa_db.sh [uniref50|uniref90|uniref30] [DEST]
set -euo pipefail

DB="${1:-uniref50}"
DEST="${2:-data/msa_db}"
THREADS="$(nproc 2>/dev/null || echo 8)"

# INSTALLED 2026-07-16: mmseqs avx2 static build at ~/mmseqs/bin (WSL2 Ubuntu),
# on PATH via ~/.bashrc. No sudo, no Docker. Functionally smoke-tested.
#
# WSL NETWORKING IS BROKEN ON THIS HOST: DNS resolves but ALL outbound HTTPS from
# inside WSL times out (verified against github.com, mmseqs.com, ftp.uniprot.org).
# So the mmseqs binary AND the uniref50 fasta had to be downloaded on the WINDOWS
# side and handed to WSL via /mnt/c. Consequences encoded below:
#   * This script must NEVER curl from inside WSL -- it will hang.
#   * The fasta is read from /mnt/c (Windows-downloaded) but the DB is BUILT ON
#     ext4 (~/msa_db): /mnt/c is slow for random IO and mmseqs search hammers it.
command -v mmseqs >/dev/null 2>&1 || {
  cat <<'EOF'
ERROR: mmseqs not found on PATH.

It was installed at ~/mmseqs/bin (WSL2 Ubuntu, avx2 static build). Put it back
on PATH:
    export PATH=$HOME/mmseqs/bin:$PATH

If it is genuinely gone, WSL outbound network is dead on this host, so DO NOT
curl from inside WSL. Download on the Windows side and copy in:
    # (Windows) curl -fsSL -o mmseqs.tgz \
    #   https://github.com/soedinglab/MMseqs2/releases/latest/download/mmseqs-linux-avx2.tar.gz
    # (WSL) tar xzf /mnt/c/path/to/mmseqs.tgz -C ~ && export PATH=$HOME/mmseqs/bin:$PATH
EOF
  exit 1
}

mkdir -p "$DEST"
avail_gb=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc '0-9')
echo "Destination: $DEST (${avail_gb} GB free), threads=${THREADS}, db=${DB}"

need_gb() {
  if [ "$avail_gb" -lt "$1" ]; then
    echo "ERROR: need ~${1} GB free, have ${avail_gb} GB. Pick a smaller db." >&2
    exit 1
  fi
}

case "$DB" in
  uniref50) URL="https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"; need_gb 80  ;;
  uniref90) URL="https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz"; need_gb 200 ;;
  uniref30)
    echo "WARNING: uniref30 is 99.4 GB DOWNLOADED and its EXTRACTED size is unverified."
    echo "         ColabFold states ~1 TB for uniref30+envdb. With ${avail_gb} GB free this"
    echo "         may not fit. Streaming the extract (no tar on disk) is the only way it"
    echo "         might. Proceeding at your own risk."
    need_gb 230
    URL="https://opendata.mmseqs.org/colabfold/uniref30_2302.db.tar.gz"
    ;;
  *) echo "unknown db '$DB' (use uniref50|uniref90|uniref30)" >&2; exit 1 ;;
esac

FASTA="$DEST/${DB}.fasta"
DBPATH="$DEST/${DB}_db"

if [ "$DB" = "uniref30" ]; then
  # Prebuilt MMseqs db: stream-extract so the 99.4 GB tar never lands on disk.
  echo "Streaming + extracting $URL ..."
  curl -sSL "$URL" | tar xzf - -C "$DEST"
  echo "NOTE: apply the taxonomy update (uniref30_2302_newtaxonomy.tar.gz, 1.8 GB)"
  echo "      -- without taxonomy this db cannot pair."
else
  # Prefer a copy already fetched by download_datasets.py --phase2 rather than
  # pulling 8.8 GB again.
  if [ ! -f "$FASTA" ]; then
    if [ -f "${FASTA}.gz" ]; then
      echo "Using already-downloaded ${FASTA}.gz (decompressing)..."
      gzip -dc "${FASTA}.gz" > "$FASTA"
    else
      echo "Downloading $URL ..."
      curl -sSL "$URL" | gzip -dc > "$FASTA"
    fi
  fi
  # Verify the fasta decompressed FULLY before createdb. HARD-WON: reading the
  # .gz off the slow /mnt/c and starting createdb too early produced a db whose
  # HEADER file (${DB}_db_h) was 0 bytes -- search still ran but convertalis /
  # result2msa died ("Invalid database read for ..._h"), i.e. NO taxonomy, i.e.
  # no pairing. The failure was silent until the a3m step. So: count seqs first.
  nseq=$(grep -c '^>' "$FASTA")
  echo "fasta: $(du -h "$FASTA" | cut -f1), ${nseq} sequences"
  [ "$nseq" -gt 1000000 ] || { echo "ERROR: only ${nseq} seqs -- fasta looks truncated" >&2; exit 1; }

  # UniRef fasta headers carry 'TaxID=<taxid>' (NOT UniProtKB's 'OX='), which
  # a3m.parse_taxid reads. VERIFIED on the real download: 50,000/50,000 headers
  # parsed (100%). Both spellings are accepted by the parser.
  echo "Building MMseqs db (createdb) ..."
  mmseqs createdb "$FASTA" "$DBPATH"

  # Verify BOTH the sequence db AND the header db (taxonomy) are non-empty. An
  # empty ${DB}_db_h is the exact corruption above and must fail loudly here,
  # not three steps later at the a3m.
  for suffix in "" "_h"; do
    f="${DBPATH}${suffix}"
    s=$(stat -c%s "$f" 2>/dev/null || echo 0)
    echo "  ${f}: ${s} bytes"
    [ "$s" -gt 0 ] || { echo "ERROR: ${f} is empty -- createdb corrupt, rerun" >&2; exit 1; }
  done
  echo "Freeing the raw fasta (the db supersedes it) ..."
  rm -f "$FASTA"
fi

cat <<EOF

DB ready at: $DEST

Next:
  1) Build the deduped query set (4,750 chains for clean_le600):
       uv run python scripts/data/prepare_msa_chains.py \\
           --split data/processed/splits/clean_le600.json \\
           --out data/processed/msa/chains_le600.fasta \\
           --index-out data/processed/msa/index_le600.json

  2) Search (ONE batched search for all queries -- not one per chain):
       mmseqs createdb data/processed/msa/chains_le600.fasta qdb
       mmseqs search qdb ${DBPATH} res tmp --threads ${THREADS} -s 5.7 --db-load-mode 0
       mmseqs result2msa qdb ${DBPATH} res msa_db --msa-format-mode 5

     --db-load-mode 0 keeps the db on disk: with 31.7 GB RAM it will NOT be
     resident. Correct but slow -- expect hours. This is expected, not a bug.

  3) Step 0 first, always:
       uv run python scripts/data/measure_msa_depth.py --a3m-dir <dir> --split ...
EOF
