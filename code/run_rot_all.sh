export OMP_NUM_THREADS=1
cd "$(dirname "$0")"
for k in 1 2 3 4 5 6 7 8 9 10; do python3 run_rotation.py --test_fold $k --B 300 > ../results/rot_$k.log 2>&1; done
