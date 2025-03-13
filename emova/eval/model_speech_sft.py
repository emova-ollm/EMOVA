import argparse
import itertools
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from emova.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from emova.conversation import conv_templates, SeparatorStyle
from emova.model.builder import load_pretrained_model
from emova.utils import disable_torch_init, read_config
from emova.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from datasets import load_dataset

from PIL import Image
import math
from functools import partial

try:
    import torch_npu
    from torch_npu.npu import amp
    from torch_npu.contrib import transfer_to_npu 
    print('Successful import torch_npu')
except Exception as e:
    print(e)


def collate_fn(batches, tokenizer):
    
    questions_ids = [_['id'] for _ in batches]
    prompts = [_['prompt'] for _ in batches]
    input_ids = []
    for z in prompts:
        input_ids_ = tokenizer_image_token(z, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids.append(input_ids_.reshape(1,-1))

    return torch.cat(input_ids,0), prompts, questions_ids

class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, data_subset, tokenizer, conv_mode):
        self.questions = load_dataset("Emova-ollm/emova-sft-speech-eval", data_subset)['test']
        self.tokenizer = tokenizer
        self.conv_mode = conv_mode
        
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        line = self.questions[idx]

        idx = line["id"]
        prompt = line['conversations'][0]['value']

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        return {
            'id': idx,
            'prompt': prompt,
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-subset', type=str, default='datasetname')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--model-path', type=str, default='/path/to/model')
    parser.add_argument('--config', type=str, default='/path/to/config')
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llama3")
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--do-sample', action='store_true')
    
    args = parser.parse_args()


    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    device = f'cuda:{local_rank}'

    model_name = get_model_name_from_path(args.model_path)

    config = read_config(args.config)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device=device, config=config
    )

    dataset = SpeechDataset(data_subset=args.data_subset, tokenizer=tokenizer, conv_mode=args.conv_mode)
    model.generation_config.pad_token_id = tokenizer.pad_token_id 
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=1,   # can only set to 1
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )
    outputs = []
    for _, (input_ids, prompts, questions_ids) in tqdm(enumerate(dataloader), total = len(dataloader)):
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids.to(device=device, non_blocking=True),
                attention_mask=torch.ones_like(input_ids).to(device),
                images=None,
                image_sizes=None,
                do_sample=args.do_sample,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                num_beams=args.num_beams,
            )
        
        result = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        outputs.append({
                'id': questions_ids[0],
                'prompt': prompts[0],
                'text': result,
            })

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, outputs)
    outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]
        
    if torch.distributed.get_rank() == 0:
        # Get the directory from the file path
        directory = os.path.dirname(args.answers_file)
        # Check if the directory exists
        if not os.path.exists(directory):
            # Create the directories if they don't exist
            os.makedirs(directory)
        with open(args.answers_file, 'w') as f:
            to_write = ''
            for line in outputs:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        