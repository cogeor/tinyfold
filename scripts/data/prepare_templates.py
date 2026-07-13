"""C6: precompute + cache retrieved monomer templates (mirrors the ESM cache).

For each query sample, retrieve a homologous monomer fold per chain
(retriever.py), copy the source backbone onto the query residues via the
alignment, and save data/processed/templates/<sample_id>.npz with:
  * template_coords_res: float32 [n_res, 4, 3]  (RAW Angstroms; the loader
    divides by global_scale, same as coords)
  * template_mask:       bool    [n_res]        (covered query residues)

frame_id is not stored (it is just chain_id at load time). Coordinates are
relative-feature inputs, so their absolute frame/translation is irrelevant; only
scale matters, hence raw Angstroms normalized at load.

Usage:
    uv run python scripts/data/prepare_templates.py \
        --clusters data/processed/clusters.json \
        --split data/processed/splits/clean_le240.json \
        --out-dir data/processed/templates
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tinyfold.retrieval.retriever import ChainEntry, MonomerRetriever


def build_retriever(parquet, clusters):
    """Index every chain; also return per-sample (seq, chain_id, row) metadata."""
    pf = pq.ParquetFile(parquet)
    retr = MonomerRetriever(k=3)
    sample_to_chain_clusters = clusters["sample_to_chain_clusters"]
    sample_to_cluster = clusters["sample_to_cluster"]
    meta = {}  # sample_id -> {"chain_id": [...], "row": int}
    row = 0
    for batch in pf.iter_batches(columns=["sample_id", "seq", "chain_id_res"], batch_size=2048):
        sids = batch.column("sample_id").to_pylist()
        seqs = batch.column("seq").to_pylist()
        cids = batch.column("chain_id_res").to_pylist()
        for sid, seq, cid in zip(sids, seqs, cids):
            cc = sample_to_cluster.get(sid)
            chain_clusters = sample_to_chain_clusters.get(sid, [None, None])
            for chain in (0, 1):
                cseq = tuple(s for s, c in zip(seq, cid) if c == chain)
                if not cseq:
                    continue
                retr.add_chain(ChainEntry(
                    sample_id=sid, chain=chain, seq=cseq,
                    chain_cluster=chain_clusters[chain], complex_cluster=cc,
                ))
            meta[sid] = {"chain_id": cid, "row": row}
            row += 1
    return retr, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/processed/samples.parquet")
    ap.add_argument("--clusters", default="data/processed/clusters.json")
    ap.add_argument("--split", default=None,
                    help="Optional split JSON; restrict query samples to its "
                         "train+test ids (else all samples).")
    ap.add_argument("--out-dir", default="data/processed/templates")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N queries")
    args = ap.parse_args()

    clusters = json.load(open(args.clusters))
    print("Building retriever index over all chains ...")
    retr, meta = build_retriever(args.parquet, clusters)
    print(f"  indexed {len(retr.entries)} chains from {len(meta)} samples")

    # Query set.
    if args.split:
        sp = json.load(open(args.split))
        query_ids = list(dict.fromkeys(sp.get("train_ids", []) + sp.get("test_ids", [])))
    else:
        query_ids = list(meta.keys())
    if args.limit:
        query_ids = query_ids[:args.limit]
    print(f"  {len(query_ids)} query samples")

    # Retrieve per query; collect needed source rows.
    retrieval = {}       # qid -> {chain: ChainTemplate or None}
    needed_rows = set()
    n_cov_chains = 0
    n_chains = 0
    for qid in query_ids:
        res = retr.retrieve_sample(qid)
        retrieval[qid] = res
        for chain, tmpl in res.items():
            n_chains += 1
            if tmpl is not None:
                n_cov_chains += 1
                needed_rows.add(meta[tmpl.src_sample_id]["row"])
    print(f"  chain coverage: {n_cov_chains}/{n_chains} "
          f"({100*n_cov_chains/max(n_chains,1):.1f}%)")

    # Read coords for needed source rows (chunked to bound RAM).
    print(f"  loading coords for {len(needed_rows)} source samples ...")
    pf = pq.ParquetFile(args.parquet)
    src_coords = {}   # row -> {chain: np.ndarray [nres_chain, 4, 3]}
    row = 0
    for batch in pf.iter_batches(columns=["atom_coords", "chain_id_res"], batch_size=1024):
        coords_col = batch.column("atom_coords").to_pylist()
        cid_col = batch.column("chain_id_res").to_pylist()
        for coords_flat, cid in zip(coords_col, cid_col):
            if row in needed_rows:
                arr = np.asarray(coords_flat, dtype=np.float32).reshape(-1, 4, 3)
                by_chain = {}
                cid_arr = np.asarray(cid)
                for chain in (0, 1):
                    sel = cid_arr == chain
                    if sel.any():
                        by_chain[chain] = arr[sel]
                src_coords[row] = by_chain
            row += 1

    # Assemble + save per query.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for qid in query_ids:
        cid = np.asarray(meta[qid]["chain_id"])
        n_res = len(cid)
        tmpl_coords = np.zeros((n_res, 4, 3), dtype=np.float32)
        tmpl_mask = np.zeros((n_res,), dtype=bool)
        # Query residue positions per chain (in the concatenated order).
        for chain in (0, 1):
            q_positions = np.where(cid == chain)[0]
            tmpl = retrieval[qid].get(chain)
            if tmpl is None or len(q_positions) == 0:
                continue
            src_row = meta[tmpl.src_sample_id]["row"]
            src_chain_coords = src_coords.get(src_row, {}).get(tmpl.src_chain)
            if src_chain_coords is None:
                continue
            for local_q, src_i in enumerate(tmpl.query_to_src):
                if src_i < 0 or src_i >= len(src_chain_coords):
                    continue
                tmpl_coords[q_positions[local_q]] = src_chain_coords[src_i]
                tmpl_mask[q_positions[local_q]] = True
        np.savez_compressed(out_dir / f"{qid}.npz",
                            template_coords_res=tmpl_coords, template_mask=tmpl_mask)
        n_written += 1
    # Residue-level coverage summary.
    print(f"Wrote {n_written} template npz files to {out_dir}")


if __name__ == "__main__":
    main()
