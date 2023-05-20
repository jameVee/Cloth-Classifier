conda deactivate
conda create -y -n fashion python=3.9.13
conda activate fashion
pip install -r requirements.txt
conda install -y -c conda-forge tensorflow
