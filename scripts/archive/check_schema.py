import pyarrow.parquet as pq
t = pq.read_table('data/processed/samples.parquet')
print(t.schema)
print(f"\nColumn names: {t.column_names}")
