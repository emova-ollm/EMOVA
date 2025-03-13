import argparse
import json
import os
import editdistance
from datasets import load_dataset

def calculate_WER(recognized_text_list, groundtruth_text_list):
    word_num = 0.0
    scores = 0.0
    for recognized_text, groundtruth_text in zip(recognized_text_list, groundtruth_text_list):
        if len(recognized_text) > 1000:
            print(recognized_text)
            continue
        recognized_word_list = recognized_text.split()
        groundtruth_word_list = groundtruth_text.split()
        current_word_num = len(groundtruth_word_list)
        word_num += current_word_num
        # Compute Levenstein's distance
        current_score = editdistance.eval(recognized_word_list, groundtruth_word_list)
        scores += current_score
    WER = scores / word_num
    return WER, scores, word_num

def calculate_CER(recognized_text_list, groundtruth_text_list):
    character_num = 0.0
    scores = 0.0
    for recognized_text, groundtruth_text in zip(recognized_text_list, groundtruth_text_list):
        if len(recognized_text) > 1000:
            print(recognized_text)
            continue
        recognized_character_list = list(recognized_text)
        groundtruth_character_list = list(groundtruth_text)
        current_character_num = len(groundtruth_character_list)
        character_num += current_character_num
        # Compute Levenstein's distance
        current_score = editdistance.eval(recognized_character_list, groundtruth_character_list)
        scores += current_score
    CER = scores / character_num
    return CER, scores, character_num

def form_ann_rst_list(ann, results, key):

    ann_dict = {}
    for item in ann:
        if key in item['id']:
            ann_dict[item['id']] = item['conversations'][-1]['value']
    
    rst_dict = {}
    for item in results:
        if key in item['id']:
            rst_dict[item['id']] = item['text']
    
    return ann_dict, rst_dict

def evaluate(args):
    results = [json.loads(q) for q in open(os.path.expanduser(args.result_path), "r")]
    ann = load_dataset("emova-asr-tts-eval/librispeech-asr-tts/")['test']

    if args.language == 'en':
        ann_dict_en, rst_dict_en = form_ann_rst_list(ann, results, 'asr')
        wer, scores_wer, word_num_wer = calculate_WER(rst_dict_en.values(), ann_dict_en.values())
        print(f'wer: {wer}, scores_wer: {scores_wer}, word_num_wer: {word_num_wer}')
    elif args.language == 'ch':
        ann_dict_ch, rst_dict_ch = form_ann_rst_list(ann, results, 'asr')
        cer, scores_cer, word_num_cer = calculate_CER(rst_dict_ch.values(), ann_dict_ch.values())
        print(f'cer: {cer}, scores_cer: {scores_cer}, word_num_cer: {word_num_cer}')
    else:
        raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", type=str, default=None)
    parser.add_argument("--language", type=str, default='en')
    args = parser.parse_args()

    evaluate(args)