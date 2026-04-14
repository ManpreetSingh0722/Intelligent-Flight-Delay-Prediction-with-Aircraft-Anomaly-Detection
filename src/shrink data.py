import pandas as pd

print("Loading... this will take a few minutes")

cols = ['AIRLINE', 'ORIGIN', 'DEST', 'DEP_TIME',
        'ARR_DELAY', 'DELAY_DUE_WEATHER', 'DELAY_DUE_CARRIER', 'DELAY_DUE_NAS', 'DELAY_DUE_LATE_AIRCRAFT']

# Read in chunks to avoid RAM crash
chunks = []
for chunk in pd.read_csv('data/raw/flights_sample_3m.csv', usecols=cols, chunksize=500000, low_memory=False):
    chunks.append(chunk)
    print(f"  Loaded {len(chunks) * 500000:,} rows so far...")

df = pd.concat(chunks)
print(f"Full dataset: {len(df):,} rows")

# Sample exactly 3 million rows
df = df.sample(n=3_000_000, random_state=42)
print(f"After sampling: {len(df):,} rows")

df.to_csv('data/flights_3m.csv', index=False)
print("Saved as data/flights_3m.csv ✓")