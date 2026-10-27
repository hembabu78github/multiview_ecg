"""Download ONLY the 100 Hz PTB-XL records needed for the study and pack them into one file.
Usage:  pip install wfdb numpy pandas
        python make_waveforms.py cohort_ids.csv ptbxl_waveforms_100hz.npz
Streams each record from PhysioNet (project ptb-xl 1.0.3). Resumable: re-run to continue.
Expected output: ~11,600 x 1000 x 12 int16 array (~280 MB uncompressed, less compressed)."""
import sys, os, numpy as np, pandas as pd, wfdb
from concurrent.futures import ThreadPoolExecutor
ids = pd.read_csv(sys.argv[1]); out = sys.argv[2]
cache = out + ".partial.npz"
done = {}
if os.path.exists(cache):
    z = np.load(cache); done = {int(k): z[k] for k in z.files}
def get(row):
    e, fn = int(row.ecg_id), row.filename_lr            # e.g. records100/00000/00001_lr
    d, f = os.path.split(fn)
    for attempt in range(4):
        try:
            sig, meta = wfdb.rdsamp(f, pn_dir=f"ptb-xl/1.0.3/{d}")
            assert sig.shape == (1000, 12), sig.shape
            return e, np.round(sig * 1000).astype(np.int16)   # microvolts, int16
        except Exception as ex:
            err = ex
    print("FAILED", e, err); return e, None
todo = [r for r in ids.itertuples() if int(r.ecg_id) not in done]
print(len(done), "already done;", len(todo), "to download")
with ThreadPoolExecutor(8) as ex:
    for i, (e, a) in enumerate(ex.map(get, todo), 1):
        if a is not None: done[e] = a
        if i % 500 == 0:
            np.savez(cache, **{str(k): v for k, v in done.items()}); print(i, "of", len(todo), flush=True)
missing = [int(r.ecg_id) for r in ids.itertuples() if int(r.ecg_id) not in done]
print("missing:", len(missing))
order = [int(r.ecg_id) for r in ids.itertuples() if int(r.ecg_id) in done]
np.savez_compressed(out, ecg_id=np.array(order), X=np.stack([done[k] for k in order]),
                    leads=np.array(["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]), unit="uV", fs=100)
print("wrote", out)
