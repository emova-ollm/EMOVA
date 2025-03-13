import argparse
import json
import random
import torch
from tqdm import tqdm
import shutil
import sys
import os

from datasets import load_dataset
from transformers import AutoModel

from emova.eval.whisper_asr.whisper_asr import load_whisper_model, EN_ASR_WER
from emova.eval.whisper_asr.paraformer_zh_asr import load_paraformer_zh_model, CH_ASR_CER
from emova.eval.whisper_asr.paraformer_zh_asr import recognize as recognize_zh
from emova.eval.whisper_asr.whisper_asr import recognize

import warnings
# Suppress FutureWarnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


def get_text_unit_list(data_file):
    text_list, unit_seq_list, id_list = [], [], []
    with open(data_file, 'r', encoding='utf-8') as f:
        line_list = [line.replace('\n', '') for line in f]
    for line in line_list:
        sample_dict = json.loads(line)
        text_list.append(sample_dict['input'])
        unit_seq_list.append(sample_dict['output'])
        id_list.append(sample_dict['id'])
    return id_list, text_list, unit_seq_list




def evaluate(args):
    speech_tokenizer = AutoModel.from_pretrained("Emova-ollm/emova_speech_tokenizer_hf", torch_dtype=torch.float32, trust_remote_code=True).eval()
    speech_tokenizer = speech_tokenizer.cuda()

    
    if args.eval_zh:
        pipe_zh = load_paraformer_zh_model(args.model_path, args.device)
    else:
        pipe_en = load_whisper_model(args.model_path, args.device)

    id_list, text_list, unit_seq_list = get_text_unit_list(args.file_path)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    wav_list = []
    for i, (text, unit) in enumerate(zip(text_list, unit_seq_list)):
        content_unit = unit.replace('<|speech_', '').replace('|>', ' ').strip()
        output_wav_file = f'{args.result_dir}/{i}.wav'
        speech_tokenizer.decode(content_unit, condition=None, output_wav_file=output_wav_file)
        wav_list.append(output_wav_file)
        

    batch_size = 12

    if args.eval_zh:
        pred_texts = recognize_zh(wav_list, text_list, pipe_zh, batch_size, False)
    else:
        pred_texts = recognize(wav_list, text_list, pipe_en, batch_size, False)



    id2ann = {}
    gpt_eval = []
    ann = load_dataset("Emova-ollm/emova-sft-speech-eval", args.data_subset)['test']

    for item in ann:
        id2ann[item['id']] = item['conversations'][1]['value']

    for idd, text, pred in zip(id_list, text_list, pred_texts):
        ann_dict = json.loads(id2ann[idd])
        new_item = {
            'id': idd,
            'gt_input': ann_dict['user question text'],
            'gt_output': ann_dict['assistant response text'],
            'pred_output': pred,
        }
        gpt_eval.append(new_item)

    
    with open(os.path.splitext(args.file_path)[0] + '_tts_asr.json', 'w', encoding='utf-8') as f:
        for line in gpt_eval:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    # shutil.rmtree(args.result_dir)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path', type=str, default='/path/to/text_unit')
    parser.add_argument('--model-path', type=str, default='/path/to/asr_model')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--data-subset', type=str, default='datasetname')
    parser.add_argument('--result-dir', type=str, default='/path/to/generated/wav')
    parser.add_argument('--eval-zh', action='store_true')
    args = parser.parse_args()

    evaluate(args)


    