import pyarrow.parquet as pq

files = [
    'data/all_events_combined_2015_2016.parquet',
    'data/xt_events_combined_2015_2016.parquet',
]

for fname in files:
    try:
        pf = pq.read_table(fname, memory_map=True)
        print(f"\n{'='*70}")
        print(f"FILE: {fname}")
        print(f"{'='*70}")
        print(f"Columns ({len(pf.column_names)}):")
        for col in sorted(pf.column_names):
            print(f"  - {col}")
        
        # Check for carry columns
        carry_cols = [c for c in pf.column_names if 'carry' in c.lower()]
        if carry_cols:
            print(f"\nCARRY-RELATED COLUMNS FOUND:")
            for col in carry_cols:
                print(f"  - {col}")
        else:
            print(f"\nNO CARRY-RELATED COLUMNS")
    except Exception as e:
        print(f"Error reading {fname}: {e}")
