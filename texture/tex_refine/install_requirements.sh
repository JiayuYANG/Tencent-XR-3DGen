# nvdiffrast
sudo apt-get update
sudo apt-get install --no-install-recommends \
pkg-config \
libglvnd0 \
libgl1 \
libglx0 \
libegl1 \
libgles2 \
libglvnd-dev \
libgl1-mesa-dev \
libegl1-mesa-dev \
libgles2-mesa-dev \
cmake \
curl

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics 
export PYOPENGL_PLATFORM=egl
sudo cp /home/ubuntu/efs/external/nvdiffrast/docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

pip install ninja imageio imageio-ffmpeg

cd /home/ubuntu/efs/external/nvdiffrast
pip install .
cd -

pip install opencv-python, rembg, dotmap, onnxruntime, segment_anything, cupy-cuda12, basics, realesrgan, compel, kiui, peft

# Diffusers
pip install diffusers["torch"] transformers

# Fix the bug in basics
vim /opt/pytorch/lib/python3.12/site-packages/basics/data/degradations-py
(Change functional_tensor to functional)

# xformers
pip3 install -U xformers -index-url https:/download.pytorch.org/whl/cu126

# install ip-adapter
pip install git+https://github.com/tencent-ailab/IP-Adapter.git
