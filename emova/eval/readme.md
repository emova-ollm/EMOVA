# EMOVA Evaluation

> [!NOTE]  
> Throughout this whole document, we should use the `Origin Format` checkpoints in [Model Zoo](./../../README.md#model-zoo).

## Evaluation on Speech Benchmarks

### Introduction

We evaluate EMOVA on ASR, TTS and speech dialogue tasks. To start, please download the EMOVA model and its corresponding config and set `MODEL_PATH` and `CONFIG_PATH`. 

Prepare evaluation data in `DATA_ROOT` as it will also save certain intermediate outputs. All the audio files generated during evaluation will be saved to `TMP_DIR`.  

1. Install [emova_speech_tokenizer](https://github.com/emova-ollm/EMOVA_speech_tokenizer)  
2. Prepare the following variables.

```bash
MODEL_PATH=path/to/model/ckpt
CONFIG_PATH=path/to/model/config
DATA_ROOT=path/to/data/root
TMP_DIR=/path/to/cache
OPENAI_API_KEY=your_openai_key
```

### ASR/TTS
```bash
python -m torch.distributed.launch \
    --nproc_per_node 8 \
    --master_addr $main_ip \
    --nnodes $num_machines \
    --node_rank $machine_rank \
    --master_port $main_port \
    --use_env \
    emova/eval/model_speech_asr_tts.py \
    --model-path $MODEL_PATH \
    --config $CONFIG_PATH \
    --answers-file ${DATA_ROOT}/librispeech_result.json \
    --temperature 1 \
    --num-workers 4 \
    --conv-mode qwen2 \

if [ "$machine_rank" -eq 0 ]; then
python emova/eval/eval_wer_cer.py \
    --result-path ${DATA_ROOT}/librispeech_result.json \
    --language en
python emova/eval/extract_tts.py --result-path  ${DATA_ROOT}/librispeech_result.json_result.json
python emova/eval/eval_tts.py --file-path ${DATA_ROOT}/librispeech_result_tts_only.json --result-dir $TMP_DIR --model-path openai/whisper-large-v3
fi
```


### Speech Dialogue

```bash
for DATA_SUBSET in emova-speech-text-en emova-speech-text-zh; do

    python -m torch.distributed.launch \
        --nproc_per_node 8 \
        --master_addr $main_ip \
        --nnodes $num_machines \
        --node_rank $machine_rank \
        --master_port $main_port \
        --use_env \
        emova/eval/model_speech_sft.py \
        --model-path $MODEL_PATH \
        --config $CONFIG_PATH \
        --data-subset $DATA_SUBSET \
        --answers-file ${DATA_ROOT}/${DATA_SUBSET}_result.json \
        --temperature 1 \
        --num-workers 4 \
        --max_new_tokens 4096 \
        --conv-mode qwen2 

    if [ "$machine_rank" -eq 0 ]; then
        python emova/eval/extract_speech_json.py --result-path ${DATA_ROOT}/${DATA_SUBSET}_result.json --data-subset $DATA_SUBSET
        python emova/eval/gpt_eval_speech.py --input_file_path  ${DATA_ROOT}/${DATA_SUBSET}_result_gpt_score.json
        python emova/eval/gpt_eval_speech_cls.py --input_file_path ${DATA_ROOT}/${DATA_SUBSET}_result_gpt_score_cls.json
        if [ "$DATA_SUBSET" == "emova-speech-text-zh" ]; then
            python emova/eval/eval_end2end.py --file-path ${DATA_ROOT}/${DATA_SUBSET}_result_tts_only.json --result-dir $TMP_DIR --model-path paraformer-zh --data-subset $DATA_SUBSET --eval-zh
        else
            python emova/eval/eval_end2end.py --file-path ${DATA_ROOT}/${DATA_SUBSET}_result_tts_only.json --result-dir $TMP_DIR --model-path openai/whisper-large-v3 --data-subset $DATA_SUBSET
        fi
        python emova/eval/gpt_eval_speech_no_punc.py --input_file_path ${DATA_ROOT}/${DATA_SUBSET}_result_tts_only_tts_asr.json
    fi
done
```

### Speech Dialogue with Images
```bash
for DATA_SUBSET in emova-speech-image-en emova-speech-image-zh; do

    python -m torch.distributed.launch \
        --nproc_per_node 8 \
        --master_addr $main_ip \
        --nnodes $num_machines \
        --node_rank $machine_rank \
        --master_port $main_port \
        --use_env \
        emova/eval/model_speech_sft_img.py \
        --model-path $MODEL_PATH \
        --config $CONFIG_PATH \
        --data-subset $DATA_SUBSET \
        --answers-file ${DATA_ROOT}/${DATA_SUBSET}_result.json \
        --temperature 1 \
        --num-workers 4 \
        --max_new_tokens 4096 \
        --conv-mode qwen2 

    if [ "$machine_rank" -eq 0 ]; then
        python emova/eval/extract_speech_json.py --result-path ${DATA_ROOT}/${DATA_SUBSET}_result.json --data-subset $DATA_SUBSET
        python emova/eval/gpt_eval_speech.py --input_file_path  ${DATA_ROOT}/${DATA_SUBSET}_result_gpt_score.json
        python emova/eval/gpt_eval_speech_cls.py --input_file_path ${DATA_ROOT}/${DATA_SUBSET}_result_gpt_score_cls.json
        if [ "$DATA_SUBSET" == "emova-speech-image-zh" ]; then
            python emova/eval/eval_end2end.py --file-path ${DATA_ROOT}/${DATA_SUBSET}_result_tts_only.json --result-dir $TMP_DIR --model-path paraformer-zh --data-subset $DATA_SUBSET --eval-zh
        else
            python emova/eval/eval_end2end.py --file-path ${DATA_ROOT}/${DATA_SUBSET}_result_tts_only.json --result-dir $TMP_DIR --model-path openai/whisper-large-v3 --data-subset $DATA_SUBSET
        fi
        python emova/eval/gpt_eval_speech_no_punc.py --input_file_path ${DATA_ROOT}/${DATA_SUBSET}_result_tts_only_tts_asr.json
    fi
done
```

## Evaluation on Vision-Language Benchmarks

We adopt the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) toolbox for evaluating EMOVA's performance on vision-language benchmarks. The detailed instructions are coming soon!
