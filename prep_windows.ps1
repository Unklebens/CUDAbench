python -m venv bench_venv
.\bench_venv\Scripts\Activate.ps1  # PowerShell
# or .\bench_venv\Scripts\activate.bat  # CMD

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cu121
