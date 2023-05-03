#!/bin/sh
cd ..
python -m debugpy --listen localhost:13840 --wait-for-client System/main.py --post \
-output "/home/bayehez2/results/bashan/remote_debug" \
-organized-folder "/home/bayehez2/groupData/organized_data/bashan/remote_debug" \
-tumor "A375 skin malignant melanoma" -pert geldanamycin \
-start 0 -end 1 -test-num 0  \
--drug-combinations
