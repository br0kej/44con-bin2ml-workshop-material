#!/bin/bash

# Setup bin2ml
echo "[-] Cloning and Building bin2ml"
git clone https://github.com/br0kej/bin2ml.git
cd bin2ml
cargo install --path .
echo "[+] Successfully built bin2ml"

# Setup python environment
echo "[-] Setting up python environment"
cd ../py/
python3 -m venv venv
source venv/bin/activate
python3 -m pip install .

echo "Successfully setup python environment. Navigate to https://drive.google.com/file/d/1dtM10-UHaG9IIAJ6KrZL8lCyHH-saZlZ/view?usp=share_link to get the data"

cd ../..
