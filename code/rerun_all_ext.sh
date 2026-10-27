export OMP_NUM_THREADS=1
cd "$(dirname "$0")"
python3 run_ext.py > ../results/ext/run_ext.log 2>&1 && python3 monitors.py > ../results/ext/monitors.log 2>&1
