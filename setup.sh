conda create -n augmented_memory python=3.10
conda activate augmented_memory
conda install -c conda-forge rdkit
conda install morfeus-ml -c conda-forge
conda install -c conda-forge openbabel
conda install -c conda-forge xtb-python
pip install reinvent-models==0.0.15rc1
conda install -c conda-forge pathos
conda install -c anaconda requests
conda install -c openeye openeye-toolkits
conda install -c conda-forge pydantic
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install dacite
conda install -c conda-forge tensorboard
conda install -c conda-forge tqdm
conda install -c conda-forge blas=1.0=mkl
cd reinvent-scoring
git checkout espsim
pip install -e .
cd ..
cd reinvent-chemistry
git checkout espsim
pip install -e .
cd ..
