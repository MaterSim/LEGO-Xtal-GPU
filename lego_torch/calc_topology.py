import argparse
import glob
import math
import os
import sqlite3
import time
from functools import partial
from multiprocessing import Pool, set_start_method
from pathlib import Path
from typing import List

from ase.db import connect
from pyxtal.db import database_topology

# ASE v9-compatible systems table (omit UNIQUE on unique_id for safety)
ASE_SYSTEMS_DDL = """CREATE TABLE systems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    unique_id TEXT,
    ctime REAL,
    mtime REAL,
    username TEXT,
    numbers BLOB,
    positions BLOB,
    cell BLOB,
    pbc INTEGER,
    initial_magmoms BLOB,
    initial_charges BLOB,
    masses BLOB,
    tags BLOB,
    momenta BLOB,
    constraints TEXT,
    calculator TEXT,
    calculator_parameters TEXT,
    energy REAL,
    free_energy REAL,
    forces BLOB,
    stress BLOB,
    dipole BLOB,
    magmoms BLOB,
    magmom REAL,
    charges BLOB,
    key_value_pairs TEXT,
    data BLOB,
    natoms INTEGER,
    fmax REAL,
    smax REAL,
    volume REAL,
    mass REAL,
    charge REAL)"""


def ensure_ase_aux_tables(db_path: str):
    """
    Ensure ASE's auxiliary tables exist in the SQLite DB.
    Some older ASE DBs only have 'systems', 'keys', and 'information'.
    Newer ASE code expects 'species', 'text_key_values', and 'number_key_values'.
    This creates the missing tables and indexes if needed.
    """
    stmts = [
        # Tables
        """
        CREATE TABLE IF NOT EXISTS species (
            Z INTEGER,
            n INTEGER,
            id INTEGER,
            FOREIGN KEY (id) REFERENCES systems(id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS text_key_values (
            key TEXT,
            value TEXT,
            id INTEGER,
            FOREIGN KEY (id) REFERENCES systems(id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS number_key_values (
            key TEXT,
            value REAL,
            id INTEGER,
            FOREIGN KEY (id) REFERENCES systems(id)
        )
        """,
        # Indexes (match ASE defaults; IF NOT EXISTS for idempotency)
        "CREATE INDEX IF NOT EXISTS species_index ON species(Z)",
        "CREATE INDEX IF NOT EXISTS key_index ON keys(key)",
        "CREATE INDEX IF NOT EXISTS text_index ON text_key_values(key)",
        "CREATE INDEX IF NOT EXISTS number_index ON number_key_values(key)",
    ]

    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        for sql in stmts:
            try:
                cur.execute(sql)
            except sqlite3.Error:
                # Be permissive: continue even if an index/table already exists
                pass
        con.commit()

def split_ase_db_by_id(in_db, out_dir="chunks", chunk_size=1000,prefix="mof"):
    print(f"\n=== SPLITTING DATABASE ===")
    t0 = time.time()
    
    os.makedirs(out_dir, exist_ok=True)

    # Inspect source once
    with sqlite3.connect(in_db) as src:
        s = src.cursor()
        tables = _list_tables(s)
        if "systems" not in tables:
            raise RuntimeError("Source is not an ASE-style DB (no 'systems' table).")
        max_id = _max_system_id(s)
        if max_id is None:
            print("No rows found in 'systems'.")
            return []
        create_sql = {t: _get_create_table_sql(s, t) for t in tables}
        # (Optional) also copy indexes/triggers
        index_sql = _get_index_sql(s, tables)
        trigger_sql = _get_trigger_sql(s, tables)

    n_chunks = math.ceil(max_id / chunk_size)
    print(f"Splitting {in_db}: max id = {max_id}, {n_chunks} chunks of {chunk_size}")

    out_paths = []
    for ci in range(n_chunks):
        lo = ci * chunk_size + 1
        hi = min((ci + 1) * chunk_size, max_id)
        out_db = os.path.join(out_dir, f"{prefix}_{ci}.db")
        if os.path.exists(out_db):
            os.remove(out_db)

        dst = sqlite3.connect(out_db)
        d = dst.cursor()
        # Speed up writes for brand-new DBs
        d.execute("PRAGMA journal_mode=MEMORY")
        d.execute("PRAGMA synchronous=OFF")
        d.execute("PRAGMA temp_store=MEMORY")
        d.execute("PRAGMA foreign_keys=OFF")
        d.execute("BEGIN")

        # Attach the source so we can read from it
        d.execute(f"ATTACH DATABASE '{in_db}' AS src")

        # 1) Recreate schemas
        for t in tables:
            if t == "systems":
                d.execute(ASE_SYSTEMS_DDL)
            else:
                d.execute(create_sql[t])

        # 2) Copy non-systems tables wholesale from src
        for t in tables:
            if t == "systems":
                continue
            d.execute(f"INSERT INTO {t} SELECT * FROM src.{t}")

        # 3) Copy only the id slice for systems
        # Use explicit column list to avoid schema-order pitfalls
        s.execute("PRAGMA table_info(systems)")
        sys_cols = [r[1] for r in s.fetchall()]
        cols_clause = ", ".join(sys_cols)
        d.execute(
            f"INSERT INTO systems({cols_clause}) "
            f"SELECT {cols_clause} FROM src.systems WHERE id BETWEEN ? AND ?",
            (lo, hi),
        )

        # 4) Recreate indexes/triggers (optional, safe if any)
        for sql in index_sql:
            d.execute(sql)
        for sql in trigger_sql:
            d.execute(sql)

        d.execute("COMMIT")
        d.execute("DETACH DATABASE src")
        dst.close()

        out_paths.append(out_db)
        print(f"Wrote {out_db} (id {lo}–{hi})")

    dt = time.time() - t0
    print(f"Database splitting completed in {dt:.2f}s")
    return out_paths

def _list_tables(cur):
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    return [r[0] for r in cur.fetchall()]

def _get_create_table_sql(cur, tname):
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (tname,))
    row = cur.fetchone()
    if row and row[0]:
        return row[0]
    # Fallback: minimal schema (rare)
    cur.execute(f"PRAGMA table_info({tname})")
    cols = [r[1] for r in cur.fetchall()]
    cols_clause = ", ".join(f'"{c}"' for c in cols)
    return f'CREATE TABLE "{tname}" ({cols_clause})'

def _get_index_sql(cur, tables):
    cur.execute("SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL")
    return [r[0] for r in cur.fetchall() if r[0] and any(f' "{t}" ' in r[0] or f' {t} ' in r[0] for t in tables)]

def _get_trigger_sql(cur, tables):
    cur.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND sql IS NOT NULL")
    return [r[0] for r in cur.fetchall() if r[0]]

def _max_system_id(cur):
    cur.execute("SELECT MAX(id) FROM systems")
    return cur.fetchone()[0]

def list_tables(cur) -> List[str]:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    return [r[0] for r in cur.fetchall()]

def list_indexes_sql(cur) -> List[str]:
    cur.execute("SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL")
    return [r[0] for r in cur.fetchall() if r[0]]

def list_triggers_sql(cur) -> List[str]:
    cur.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND sql IS NOT NULL")
    return [r[0] for r in cur.fetchall() if r[0]]

def get_create_table_sql(cur, tname: str) -> str:
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (tname,))
    row = cur.fetchone()
    if row and row[0]:
        return row[0]
    # Fallback (rare)
    cur.execute(f"PRAGMA table_info({tname})")
    cols = [r[1] for r in cur.fetchall()]
    cols_clause = ", ".join(f'"{c}"' for c in cols)
    return f'CREATE TABLE "{tname}" ({cols_clause})'

def merge_ase_chunks(chunks_dir: str, out_db: str, preserve_ids: bool = True):
    print(f"\n=== MERGING CHUNKS ===")
    t0 = time.time()
    
    chunk_paths = sorted(glob.glob(os.path.join(chunks_dir, "*.db")))
    if not chunk_paths:
        raise RuntimeError(f"No .db files found in {chunks_dir}")

    print(f"Merging {len(chunk_paths)} chunks into {out_db}")

    template_db = chunk_paths[0]
    with sqlite3.connect(template_db) as tmpl:
        tcur = tmpl.cursor()
        tables = list_tables(tcur)
        if "systems" not in tables:
            raise RuntimeError("Template DB has no 'systems' table; not an ASE DB?")
        create_sql = {t: get_create_table_sql(tcur, t) for t in tables}
        index_sql = list_indexes_sql(tcur)
        trigger_sql = list_triggers_sql(tcur)

    if os.path.exists(out_db):
        os.remove(out_db)
    out = sqlite3.connect(out_db)
    cur = out.cursor()

    # Speed + stability
    cur.execute("PRAGMA journal_mode=MEMORY")
    cur.execute("PRAGMA synchronous=OFF")
    cur.execute("PRAGMA temp_store=MEMORY")
    cur.execute("PRAGMA foreign_keys=OFF")
    cur.execute("PRAGMA busy_timeout=5000")  # wait up to 5s for any transient locks

    # 1) Create schema
    cur.execute("BEGIN")
    for t in tables:
        if t == "systems":
            cur.execute(ASE_SYSTEMS_DDL)
        else:
            cur.execute(create_sql[t])
    out.commit()  # finalize schema DDL

    # 2) Copy support tables once (from first chunk)
    cur.execute(f"ATTACH DATABASE '{template_db}' AS src0")
    cur.execute("BEGIN")
    for t in tables:
        if t == "systems":
            continue
        cur.execute(f"INSERT INTO {t} SELECT * FROM src0.{t}")
    out.commit()             # finalize reads/writes touching src0
    cur.execute("DETACH DATABASE src0")  # now safe to detach

    # 3) Append systems from all chunks (commit+detach each time)
    total_rows = 0
    for i, path in enumerate(chunk_paths):
        print(f"Processing chunk {i+1}/{len(chunk_paths)}: {path}")
        
        cur.execute(f"ATTACH DATABASE '{path}' AS src")
        cur.execute("BEGIN")
        
        if preserve_ids:
            cur.execute("INSERT INTO systems SELECT * FROM src.systems")
        else:
            # Reassign new IDs by omitting id column if present
            cur.execute("PRAGMA table_info(systems)")
            cols = [r[1] for r in cur.fetchall()]
            if "id" in cols:
                non_id = [c for c in cols if c != "id"]
                cols_clause = ", ".join(non_id)
                cur.execute(f"INSERT INTO systems({cols_clause}) SELECT {cols_clause} FROM src.systems")
            else:
                cur.execute("INSERT INTO systems SELECT * FROM src.systems")
        
        # Count rows added from this chunk
        cur.execute("SELECT COUNT(*) FROM src.systems")
        chunk_rows = cur.fetchone()[0]
        total_rows += chunk_rows
        print(f"  Added {chunk_rows} rows")
        
        out.commit()          # finalize writes that referenced 'src'
        cur.execute("DETACH DATABASE src")

    # 4) Recreate indexes/triggers on merged DB
    cur.execute("BEGIN")
    for sql in index_sql:
        try:
            cur.execute(sql)
        except sqlite3.Error as e:
            print(f"Warning: Could not recreate index: {e}")
    for sql in trigger_sql:
        try:
            cur.execute(sql)
        except sqlite3.Error as e:
            print(f"Warning: Could not recreate trigger: {e}")
    out.commit()
    out.close()

    # 5) Post-merge fixes: ensure IDs and ASE aux tables
    with sqlite3.connect(out_db) as c:
        cur2 = c.cursor()
        # If some rows have NULL id (due to missing AUTOINCREMENT schema), assign rowid
        cur2.execute("SELECT SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END) FROM systems")
        null_ids = cur2.fetchone()[0]
        if null_ids and null_ids > 0:
            print(f"Fixing {null_ids} NULL ids by assigning rowid...")
            cur2.execute("UPDATE systems SET id = rowid WHERE id IS NULL")
            c.commit()

        # Ensure aux tables exist
        ensure_ase_aux_tables(out_db)

        # Optional: compact / optimize
        cur2.execute("PRAGMA optimize")

    dt = time.time() - t0
    print(f"Merging completed in {dt:.2f}s")
    print(f"Final database: {out_db} with {total_rows} total rows")



def process_one(db_path: str, overwrite: bool = False):
    stem = Path(db_path).stem  # e.g., mof_0
    prefix_dir = Path("tmp") / stem
    prefix_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(prefix_dir / "mof")  # tmp/mof_0/mof.cif per process
    
    t0 = time.time()
    # Ensure DB schema is compatible with modern ASE writes/updates
    ensure_ase_aux_tables(db_path)
    db = database_topology(db_path)
    
    print(f"Processing {db_path} with {db.db.count()} structures...")
    
    # Update topology
    t_topo = time.time()
    db.update_row_topology(overwrite=overwrite, prefix=prefix)
    dt_topo = time.time() - t_topo
    
    # Clean structures
    t_clean = time.time()
    db.clean_structures_spg_topology(dim=3)
    dt_clean = time.time() - t_clean
    
    dt_total = time.time() - t0
    print(f"  Topology update: {dt_topo:.2f}s")
    print(f"  Structure cleaning: {dt_clean:.2f}s")
    
    return db_path, dt_total, dt_topo, dt_clean

def run_topology_pipeline(
    in_db: str,
    out_dir: str = "chunks_topo",
    out_db: str = "mof-top-topology.db",
    chunk_size: int = 1000,
    nprocs: int = 96,
    prefix: str = "mof",
    overwrite: bool = False,
    warmup: bool = False,
    only_missing: bool = True,
):
    """
    Complete topology processing pipeline.
    
    Args:
        in_db: Input database path
        out_dir: Output directory for chunk databases
        out_db: Final merged database path
        chunk_size: Number of structures per chunk
        nprocs: Number of worker processes
        prefix: Filename prefix for chunks
        overwrite: Overwrite existing topology fields
        warmup: Run warmup on first chunk
    """
    total_start = time.time()
    
    # Step 1: Split database into chunks
    print(f"\n=== STEP 1: SPLITTING DATABASE ===")
    t_split = time.time()
    chunk_paths = split_ase_db_by_id(in_db, out_dir, chunk_size, prefix)
    dt_split = time.time() - t_split
    print(f"Split completed in {dt_split:.2f}s")

    
    # Helper: check if a chunk already has topology assigned for all rows
    def _chunk_complete(path: str) -> bool:
        try:
            with sqlite3.connect(path) as con:
                cur = con.cursor()
                cur.execute("SELECT COUNT(*) FROM systems")
                total = cur.fetchone()[0] or 0
                cur.execute("SELECT COUNT(*) FROM keys WHERE key='topology'")
                have = cur.fetchone()[0] or 0
                return total > 0 and have == total
        except sqlite3.Error:
            return False

    # Step 2: Process chunks (topology calculation)
    print(f"\n=== STEP 2: PROCESSING TOPOLOGY ===")
    
    # Use spawn start method for JuliaCall stability
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass  # Already set in this interpreter

    # Optional warmup to avoid many concurrent Julia precompilations
    # Optionally skip chunks that are already complete (unless overwrite)
    if only_missing and not overwrite:
        before = len(chunk_paths)
        chunk_paths = [p for p in chunk_paths if not _chunk_complete(p)]
        skipped = before - len(chunk_paths)
        if skipped:
            print(f"Skipping {skipped} already-processed chunks (topology present for all rows)")

    # Warmup on the first remaining chunk, if any
    dt_warmup = 0.0
    if chunk_paths:
        print(f"locating {chunk_paths[0]} for warmup...")
        print(f"Warming up on {chunk_paths[0]}...")
        t_warmup = time.time()
        p, dt_total, dt_topo, dt_clean = process_one(chunk_paths[0], overwrite=overwrite)
        dt_warmup = time.time() - t_warmup
        print(f"Warmup completed in {dt_warmup:.2f}s: {p}")
        chunk_paths = chunk_paths[1:]
    else:
        print("No chunks need processing.")


    if not chunk_paths:
        print("No remaining chunks to process.")
        dt_process = 0
        total_topo_time = 0
        total_clean_time = 0
    else:
        print(f"Processing {len(chunk_paths)} chunks with {nprocs} workers...")
        t_process = time.time()
        worker = partial(process_one, overwrite=overwrite)
        
        total_topo_time = 0
        total_clean_time = 0
        
        with Pool(processes=nprocs) as pool:
            for i, (p, dt_total, dt_topo, dt_clean) in enumerate(pool.imap_unordered(worker, chunk_paths), 1):
                total_topo_time += dt_topo
                total_clean_time += dt_clean
                print(f"[{i}/{len(chunk_paths)}] {p} completed in {dt_total:.2f}s")
        
        dt_process = time.time() - t_process
        print(f"All topology processing completed in {dt_process:.2f}s")
        print(f"Total topology time: {total_topo_time:.2f}s")
        print(f"Total cleaning time: {total_clean_time:.2f}s")


    # Step 3: Merge chunks back together
    print(f"\n=== STEP 3: MERGING CHUNKS ===")
    t_merge = time.time()

    merge_ase_chunks(chunks_dir=out_dir, out_db=out_db, preserve_ids=False)
    dt_merge = time.time() - t_merge
    
    # Final summary
    total_time = time.time() - total_start
    print(f"\n=== PIPELINE COMPLETED ===")
    print(f"Step 1 - Database splitting: {dt_split:.2f}s")
    print(f"Step 2 - Topology processing:")
    print(f"  Warmup time: {dt_warmup:.2f}s")
    print(f"  Processing time: {dt_process:.2f}s")
    print(f"  Total topology calculation: {total_topo_time:.2f}s")
    print(f"  Total structure cleaning: {total_clean_time:.2f}s")
    print(f"Step 3 - Chunk merging: {dt_merge:.2f}s")
    print(f"Total pipeline time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    
    return {
        'total_time': total_time,
        'split_time': dt_split,
        'warmup_time': dt_warmup,
        'process_time': dt_process,
        'topology_time': total_topo_time,
        'clean_time': total_clean_time,
        'merge_time': dt_merge,
        'output_db': out_db
    }
def main():
    ap = argparse.ArgumentParser(description="Process MOF database topology in chunks")
    
    # Input/Output arguments
    ap.add_argument("--in-db", required=True, help="Input DB (e.g., mof-top.db)")
    ap.add_argument("--out-dir", default="chunks_topo", help="Output dir for chunk DBs")
    ap.add_argument("--out-db", default="mof-top-topology.db", help="Final merged DB path")
    
    # Processing arguments
    ap.add_argument("--chunk-size", type=int, default=1000, help="Rows per chunk DB")
    ap.add_argument("--nprocs", type=int, default=96, help="Number of worker processes")
    ap.add_argument("--prefix", default="mof", help="Base filename prefix for chunk DBs")
    
    # Operation flags
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing topology fields")
    ap.add_argument("--warmup", action="store_true", help="Run one chunk first to warm Julia precompilation")

    args = ap.parse_args()
    
    # Run the pipeline
    results = run_topology_pipeline(
        in_db=args.in_db,
        out_dir=args.out_dir,
        out_db=args.out_db,
        chunk_size=args.chunk_size,
        nprocs=args.nprocs,
        prefix=args.prefix,
        overwrite=args.overwrite,
        warmup=args.warmup
    )
    
    print(f"\nPipeline results saved to: {results['output_db']}")
    

if __name__ == "__main__":
    main()
