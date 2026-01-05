python3 -m venv bench_venv
source bench_venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install numpy
python -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu126