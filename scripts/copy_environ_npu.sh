#!/bin/bash
skip_install=false

set -e
set -x

function safe_mkdir() {
    if [ ! -d $1 ]; then
      mkdir $1
    fi
}
function mox() {
  if [ ! -d "$2" ] || [ "$3" = "True" ]; then
    python -c "import moxing as mox; mox.file.copy_parallel('s3://${BUCKET}/$1', '$2')"
  fi
}
function lns() {
  if [ ! -d $2 ]; then
    ln -s $1 $2
  fi
}

rm -rf /home/ma-user/anaconda3/envs/minigpt4o/
if [ ! -d "/home/ma-user/anaconda3/envs/minigpt4o/" ]; then
  source /home/ma-user/anaconda3/etc/profile.d/conda.sh; conda create -n minigpt4o --clone python-3.9.10
fi

source /home/ma-user/anaconda3/etc/profile.d/conda.sh; conda activate minigpt4o;
python -V

if [ "$skip_install" = false ]; then
  pip install protobuf==3.20
fi

export JAVA_HOME=/home/ma-user/modelarts/java/jdk1.8.0_301/
export PATH=${JAVA_HOME}/bin:${PATH}


if [ "$skip_install" = false ]; then
  echo "------------------installing packages----------------------"
  pip install setuptools==60
  pip install torch==2.4.0 pyyaml numpy==1.23.5 decorator scipy attrs psutil torch_npu==2.4.0 modelcards omegaconf imageio ftfy scikit-learn opencv-python datasets==2.16.1 transformers==4.41.1 tokenizers==0.19.1 sentencepiece==0.1.99 shortuuid accelerate==0.27.2 bitsandbytes pydantic==1.10.13 markdown2[all] numpy scikit-learn==1.2.2 gradio gradio_client==0.8.1 requests httpx==0.24.0 uvicorn fastapi einops==0.6.1 einops-exts==0.0.4 timm==0.6.13 mmcv==1.7.0 tensorboardX ninja wandb
  pip install peft==0.4.0 deepspeed==0.14.4
  pip install loguru
  pip install packaging

  pip install transformers==4.44.2
  pip install accelerate==0.33.0
fi

echo 'Setup environ done.'
