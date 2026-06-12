"""Check timestamp distribution of scenario10_test.parquet for warm-up planning."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import polars as pl

df = pl.read_parquet('data/processed/scenario10_test.parquet').sort('timestamp')
n = len(df)
t0 = df['timestamp'][0]
t_end = df['timestamp'][-1]
print(f'Total flows: {n:,}')
print(f'Time span: {t_end - t0:.1f}s  ({(t_end-t0)/3600:.2f} hours)')
print(f't0={t0:.1f}  t_end={t_end:.1f}')
print()

# Check timestamps at key flow indices
checkpoints = [0, 10000, 50000, 100000, 200000, 300000, 400000, 500000,
               600000, 620000, 650000, 700000, 800000, 900000, 1000000, n-1]
print(f'{"Flow idx":>10}  {"Timestamp":>12}  {"Real sec from start":>20}  {"Demo sec @factor=100":>22}  {"Label botnet%":>14}')
print('-' * 90)
for idx in checkpoints:
    if idx < n:
        ts = df['timestamp'][idx]
        elapsed = ts - t0
        demo_sec = elapsed / 100
        # Count botnet in surrounding 1000 flows
        window_start = max(0, idx-500)
        window_end = min(n, idx+500)
        window = df[window_start:window_end]
        bot_pct = (window['label'] == 'botnet').mean() * 100
        print(f'{idx:>10}  {ts:>12.1f}  {elapsed:>20.1f}  {demo_sec:>20.1f}s  {bot_pct:>12.1f}%')

print()
# Find when each botnet IP first and last appears
botnet_ips = ['147.32.84.165','147.32.84.191','147.32.84.192','147.32.84.193',
              '147.32.84.204','147.32.84.205','147.32.84.206','147.32.84.207',
              '147.32.84.208','147.32.84.209','147.32.96.69']

print('Botnet IP first/last appearance:')
for ip in botnet_ips:
    rows = df.filter((pl.col('src_ip') == ip) | (pl.col('dst_ip') == ip))
    if len(rows) > 0:
        first_ts = rows['timestamp'].min() - t0
        last_ts = rows['timestamp'].max() - t0
        count = len(rows)
        print(f'  {ip:22s}  first={first_ts:8.1f}s  last={last_ts:8.1f}s  flows={count:6,}')
