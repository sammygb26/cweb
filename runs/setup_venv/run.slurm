#!/bin/bash

#SBATCH --job-name=cweb
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=16G
#SBATCH --account=phys036784
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=test

module --force purge
module load languages/python/3.12.3

cd $HOME/cweb

python3 -m venv venv --prompt "cweb"

source venv/bin/activate

pip install -r ./requirements.txt
pip install -e .
