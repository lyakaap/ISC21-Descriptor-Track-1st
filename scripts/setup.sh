pip install llvmlite --ignore-installed
pip install --upgrade tensorflow
pip install -U typer timm tqdm comet_ml h5py pytorch_metric_learning
pip install -U augly
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install faiss-gpu  # conda install -c pytorch faiss-gpu
