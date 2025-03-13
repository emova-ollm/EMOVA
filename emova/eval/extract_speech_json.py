import json
import argparse
import editdistance
import string
import os.path as osp

from datasets import load_dataset

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def calculate_WER(recognized_text_list, groundtruth_text_list):
    word_num = 0.0
    scores = 0.0
    for recognized_text, groundtruth_text in zip(recognized_text_list, groundtruth_text_list):
        recognized_text = clean_text(recognized_text)
        groundtruth_text = clean_text(groundtruth_text)

        recognized_word_list = recognized_text.split()
        groundtruth_word_list = groundtruth_text.split()
        current_word_num = len(groundtruth_word_list)
        word_num += current_word_num
        current_score = editdistance.eval(recognized_word_list, groundtruth_word_list)
        scores += current_score
    WER = scores / word_num
    return WER, scores, word_num

def calculate_CER(recognized_text_list, groundtruth_text_list):
    character_num = 0.0
    scores = 0.0

    from emova.eval.cn_normalizer import TextNorm 
    normalizer = TextNorm(to_banjiao=False,to_upper=False,to_lower=False,remove_fillers=False,remove_erhua=False,check_chars=False,remove_space=True,cc_mode='')

    for recognized_text, groundtruth_text in zip(recognized_text_list, groundtruth_text_list):
        recognized_text = normalizer(recognized_text)
        groundtruth_text = normalizer(groundtruth_text)

        recognized_character_list = list(recognized_text)
        groundtruth_character_list = list(groundtruth_text)
        current_character_num = len(groundtruth_character_list)
        character_num += current_character_num
        # Compute Levenstein's distance
        current_score = editdistance.eval(recognized_character_list, groundtruth_character_list)
        scores += current_score
    CER = scores / character_num
    return CER, scores, character_num



def check_keys(item, mapping):
    assert mapping['human_text'] in item
    assert mapping['human_emo'] in item
    assert mapping['human_pitch'] in item
    assert mapping['gpt_text'] in item
    assert mapping['gpt_speech'] in item
    assert mapping['gpt_emo'] in item
    assert mapping['gpt_pitch'] in item

def extract(args):


    mapping = {
        'human_text': 'user question text',
        'human_pitch': 'user question pitch',
        'human_emo': 'user question emotion',
        'gpt_text': 'assistant response text',
        'gpt_emo': 'assistant response emotion',
        'gpt_pitch': 'assistant response pitch',
        'gpt_speech': 'assistant response speech',
    }

    result =  [json.loads(q) for q in open(args.result_path, "r")]
    ann = load_dataset("Emova-ollm/emova-sft-speech-eval", args.data_subset, token="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")['test']
    
    rst_dict = {}
    ann_dict = {}
    rst_failed = []
    for item in ann:
        ann_dict[item['id']] = json.loads(item['conversations'][1]['value'])
    for item in result:
        try:
            if 'pred' in item:
                rst = item['pred']
            elif 'text' in item:
                rst = item['text']

            if 'id' in item:
                idx = item['id']
            elif 'question_id' in item:
                idx = item['question_id']

            new_dict = json.loads(rst)   # text/pred
            check_keys(new_dict, mapping)
            rst_dict[idx] = new_dict  # id/question_id
        except Exception as e:
            rst_failed.append(item)
            print(e)

    print("\% success json parsing: ", 1-(len(rst_failed) / len(result)))

    inst_gt = []
    inst_pred = []

    for key, value in rst_dict.items():
        inst_gt.append(ann_dict[key][mapping['human_text']])
        inst_pred.append(value[mapping['human_text']])
    
    if 'zh' in args.data_subset:
        cer, _, _ = calculate_CER(inst_pred, inst_gt)
        print('cer (speech instruction): ', cer)
    else:
        wer, _, _ = calculate_WER(inst_pred, inst_gt)
        print('wer (speech instruction): ', wer)


    tts_rst = []
    for key, value in rst_dict.items():
        new_item = {
            'id': key,
            'input': value[mapping['gpt_text']],
            'output': value[mapping['gpt_speech']],
        }
        tts_rst.append(new_item)
    
    text_score_gpt = []
    for key, value in rst_dict.items():
        new_item = {
            'id': key,
            'gt_input': ann_dict[key][mapping['human_text']],
            'gt_output': ann_dict[key][mapping['gpt_text']],
            'pred_input': value[mapping['human_text']],
            'pred_output': value[mapping['gpt_text']],
        }
        text_score_gpt.append(new_item)


    def emo_map(emo_str):
        if emo_str in ['happy', 'surprised']:
            return 'happy'
        elif emo_str in ['sad', 'fearful']:
            return 'sad'
        elif emo_str in ['angry', 'disgusted']:
            return 'angry'
        else:
            return emo_str

    cls_score_gpt = []
    for key, value in rst_dict.items():
        new_item = {
            'id': key,
            'pred_input': value[mapping['human_text']],
            'pred_output': value[mapping['gpt_text']],
            'pred_output_emotion': emo_map(value[mapping['gpt_emo']]),
            'pred_output_pitch': value[mapping['gpt_pitch']],
            'gt_output_emotion': emo_map(ann_dict[key][mapping['gpt_emo']]),
            'gt_output_pitch': ann_dict[key][mapping['gpt_pitch']]
        }
        cls_score_gpt.append(new_item)
         
    to_write = ''
    for item in text_score_gpt:
        to_write += (json.dumps(item, ensure_ascii=False) + '\n')
    with open(osp.splitext(args.result_path)[0] + '_gpt_score.json', 'w', encoding='utf-8') as f:
        f.write(to_write)

    to_write = ''
    for item in tts_rst:
        to_write += (json.dumps(item, ensure_ascii=False) + '\n')
    with open(osp.splitext(args.result_path)[0] + '_tts_only.json', 'w', encoding='utf-8') as f:
        f.write(to_write)

    to_write = ''
    for item in cls_score_gpt:
        to_write += (json.dumps(item, ensure_ascii=False) + '\n')
    with open(osp.splitext(args.result_path)[0] + '_gpt_score_cls.json', 'w', encoding='utf-8') as f:
        f.write(to_write)
         
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", type=str, default='path/to/result/file')
    parser.add_argument('--data-subset', type=str, default='datasetname')
    args = parser.parse_args()

    extract(args)