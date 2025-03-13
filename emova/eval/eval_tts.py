import argparse
import json
import random
import torch
from tqdm import tqdm
import shutil
import sys
import os

from transformers import AutoModel

from whisper_asr.whisper_asr import load_whisper_model, EN_ASR_WER
from whisper_asr.paraformer_zh_asr import load_paraformer_zh_model, CH_ASR_CER



import warnings
# Suppress FutureWarnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


def get_text_unit_list(data_file, sample_num='all'):
    text_list, unit_seq_list = [], []
    with open(data_file, 'r', encoding='utf-8') as f:
        line_list = [line.replace('\n', '') for line in f]
    for line in line_list:
        sample_dict = json.loads(line)
        text_list.append(sample_dict['input'])
        unit_seq_list.append(sample_dict['output'])
    if sample_num != 'all':
        random.seed(1234)
        id_list = list(range(sample_num))
        random.shuffle(id_list)
        text_list = [text_list[x] for x in id_list]
        unit_seq_list = [unit_seq_list[x] for x in id_list]
    return text_list, unit_seq_list




def evaluate(args):
    speech_tokenizer = AutoModel.from_pretrained("Emova-ollm/emova_speech_tokenizer_hf", torch_dtype=torch.float32, trust_remote_code=True).eval()
    speech_tokenizer = speech_tokenizer.cuda()

    
    if args.eval_zh:
        pipe_zh = load_paraformer_zh_model(args.model_path, args.device)
    else:
        pipe_en = load_whisper_model(args.model_path, args.device)

    if args.sample_num == -1:
        sample_num = 'all'
    else:
        sample_num = args.sample_num
    text_list, unit_seq_list = get_text_unit_list(args.file_path, sample_num)
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
        CH_ASR_CER(pipe_zh, wav_list, text_list, batch_size=batch_size, print_verbose=False)
    else:
        EN_ASR_WER(pipe_en, wav_list, text_list, batch_size=batch_size, print_verbose=False)

    # shutil.rmtree(args.result_dir)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path', type=str, default='/path/to/text_unit')
    parser.add_argument('--model-path', type=str, default='/path/to/asr_model')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--sample-num', type=int, default=-1)
    parser.add_argument('--result-dir', type=str, default='/path/to/generated/wav')
    parser.add_argument('--eval-zh', action='store_true')
    args = parser.parse_args()

    evaluate(args)


    