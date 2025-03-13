import json
import argparse
import os.path as osp
from datasets import load_dataset

def extract(args):
    result =  [json.loads(q) for q in open(args.result_path, "r", encoding='utf-8')]
    annotation = load_dataset("emova-asr-tts-eval/librispeech-asr-tts/")['test']

    id2ann = {}
    for item in annotation:
        id2ann[item['id']] = item

    tts_rst = []
    for item in result:
        if 'tts' in item['id']:
            ann = id2ann[item['id']]
            tts_rst.append({
                'input': ann['conversations'][0]['value'].replace('Please synthesize the speech corresponding to the follwing text.\n', ''),
                'groundtruth': ann['conversations'][1]['value'],
                'output': item['text']
            })

    to_write = ''
    for item in tts_rst:
        to_write += (json.dumps(item, ensure_ascii=False) + '\n')

    with open(osp.splitext(args.result_path)[0] + '_tts_only.json', 'w', encoding='utf-8') as f:
        f.write(to_write)
         
    
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", type=str, default='path/to/result/file')
    args = parser.parse_args()

    extract(args)