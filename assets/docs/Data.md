# Data

## EMOVA Training Datasets

To start, install the HuggingFace `datasets` packages via pip, and store the targeted data root directory path in the environment variable `DATA_ROOT`.

```bash
export DATA_ROOT=/path/to/data

pip install datasets
```



### Stage 1: Vision-language pre-alignment

For this stage, we train with the [LLaVA-Pretrain-558K](https://arxiv.org/abs/2304.08485) dataset. Follow the [official instructions](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md#pretraining-dataset) to download the datasets and then re-organize the data structure as follows:

```
├── $DATA_ROOT
│   ├── LLaVA-Pretrain
│   │   │── blip_laion_cc_sbu_558k.json
│   │   │── images
│   │   │   │── 00000
│   │   │   │── 00001
│   │   │   │── ...
```



### Stage 2: Omni-modal text-centric alignment

For this stage, we train with the [EMOVA-alignment-7M](https://huggingface.co/datasets/Emova-ollm/emova-alignment-7m) dataset. Its data has all been deployed on HuggingFace, except the [ShareGPT4V-PT](https://arxiv.org/abs/2311.12793) dataset due to its giant size.

1. Download the ShareGPT4V-PT dataset following the [official instructions](https://github.com/ShareGPT4Omni/ShareGPT4V/blob/master/docs/Data.md#sharegpt4v-pt-dataset), and then re-organize the data structure as follows.

2. Download the remaining dataset from HuggingFace and save it on disk. We have provided a script for downloading:

    ```bash
    # by default, we download the whole dataset subsets
    python scripts/convert_hf_parquet_to_llava_format.py --dataset_path Emova-ollm/emova-alignment-7m --save_root $DATA_ROOT

    # if necessary, we can only download certain subset
    python scripts/convert_hf_parquet_to_llava_format.py --dataset_path Emova-ollm/emova-alignment-7m --save_root $DATA_ROOT --config allava-caption-part0
    ```

3. Finally, the data structure should be as follows after all files are downloaded.

    ```
    ├── $DATA_ROOT
    │   ├── LLaVA-Pretrain                                              -- Stage 1
    │   ├── emova-alignment-7m                                          -- Stage 2
    │   │   ├── annotations
    │   │   │   │── share-captioner_coco_lcs_sam_1246k_1107.json        -- Download from ShareGPT4V
    │   │   │   │── allava-caption-part0.json                           -- Download from HF
    │   │   │   │── allava-caption-part1.json                           -- Download from HF
    │   │   │   │── ...
    │   │   │── LLaVA-Pretrain                                          -- Download from ShareGPT4V
    │   │   │── coco                                                    -- Download from ShareGPT4V
    │   │   │── sam                                                     -- Download from ShareGPT4V
    │   │   │── allava-caption-part0                                    -- Download from HF
    │   │   │── allava-caption-part1                                    -- Download from HF
    │   │   │── ...
    ```



### Stage 3: Omni-modal instruction tuning

For this stage, we train with the [EMOVA-sft-4M](https://huggingface.co/datasets/Emova-ollm/emova-sft-4m) dataset, and its data has been completely deployed on HuggingFace. To facilitate research on spoken dialogue, we separately deploy a copy of EMOVA spoken dialogue SFT data in the [EMOVA-sft-speech-231k](https://huggingface.co/datasets/Emova-ollm/emova-sft-speech-231k) dataset. Follow the instruction below to download the dataset from HuggingFace, and then, the data structure should be as follows after all files are downloaded.

```bash
python scripts/convert_hf_parquet_to_llava_format.py --dataset_path Emova-ollm/emova-sft-4m --save_root $DATA_ROOT
```

```
├── $DATA_ROOT
│   ├── LLaVA-Pretrain                                              -- Stage 1
│   ├── emova-alignment-7m                                          -- Stage 2
│   ├── emova-sft-4m                                                -- Stage 3
│   │   │── annotations
│   │   │   │── ai2d-cauldron-llava-format-llavaov.json
│   │   │   │── ai2d-gpt4v-llavaov.json
│   │   │   │── ...
│   │   │   │── websrc.json
│   │   │── ai2d-cauldron-llava-format-llavaov
│   │   │── ai2d-gpt4v-llavaov
│   │   │── ...
│   │   │── websrc
```



## Custom Datasets

Our code follows the [LLaVA data format](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md#dataset-format) to organize our training data. Therefore, to train with your custom data, you need to save your training data in the **JSON** format as follows:

```json
[
  {
    "id": $UNIQUE_DATA_ID$,
    "image": $IMAGE_PATH$,
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n$QUERY_ROUND_1$"
      },
      {
        "from": "gpt",
        "value": "$REPLY_ROUND_1$"
      },
      {
        "from": "human",
        "value": "$QUERY_ROUND_2$"
      },
      {
        "from": "gpt",
        "value": "$REPLY_ROUND_2$"
      },
      ...
    ]
  },
  ...
]
```

If a query contains audio inputs, we need to first extract its speech tokens using the [EMOVA speech tokenizer](https://huggingface.co/Emova-ollm/emova_speech_tokenizer_hf), and then concatenate the EMOVA speech instruction template. An example is provided as follows:

```python
import random
from transformers import AutoModel
import torch

wav_file = "/path/to/audio"

# Step 1: extract speech tokens
model = AutoModel.from_pretrained("Emova-ollm/emova_speech_tokenizer_hf", torch_dtype=torch.float32, trust_remote_code=True).eval().cuda()
speech_unit = model.encode(wav_file)

# Step 2: concat speech instruction
chat_format = r'Please recognize the texts, emotion and pitch from the user question speech units and provide the texts, emotion, pitch and speech units for the assistant response. \nEmotion should be chosen from ["neutral", "happy", "sad", "angry", "surprised", "disgusted", "fearful"]. \nPitch should be chosen from ["low", "normal", "high"].\nYour output should be in json format.\nAn output example is:\n{"user question text": "", "user question emotion": "", "user question pitch": "", "assistant response text": "", "assistant response emotion": "", "assistant response pitch": ""，"assistant response speech": ""}\n\nuser question speech:'
query = chat_format + speech_unit
```

Finally, create a new dataset config similarly with [emova-sft-4M.py](../../configs/_base_/datasets/emova_sft_4M.py)

```python
# configs/_base_/datasets/custom_dataset.py

data_args = dict(
    data_path=[
        "$PATH_TO_JSON_1.json",
        "$PATH_TO_JSON_2.json",
    ],
    image_folder="$PATH_TO_IMAGE_ROOT",
    lazy_preprocess=True,
    is_multimodal=False,
    image_aspect_ratio='pad',
)
```