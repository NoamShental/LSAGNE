#!/bin/sh
cd ..
python -m debugpy --listen localhost:13840 --wait-for-client System/cmap_organizer.py