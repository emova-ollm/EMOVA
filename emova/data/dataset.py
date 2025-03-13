import json
import os
import random
from dataclasses import dataclass

import transformers
import torch
from PIL import Image
from torch.utils.data import Dataset

from emova.utils import rank0_print

from .preprocess import *
from ..constants import IGNORE_INDEX
from ..mm_utils import expand2square, process_anyres_image


def read_data_file(file):
    if file.endswith('.json'):
        list_data_dict = json.load(open(file, "r"))
    elif file.endswith('.jsonl'):
        list_data_dict = [json.loads(line) for line in open(file, 'r', encoding='utf-8')]
    else:
        raise RuntimeError(f"Unrecoginized file: {file}")
    return list_data_dict


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazySupervisedDataset, self).__init__()

        if isinstance(data_path, list):
            rank0_print("Load annotations from list. {}".format(data_path))
            list_data_dict = []
            for sub_data_path in data_path:
                if os.path.isdir(sub_data_path):
                    for file in os.listdir(sub_data_path):
                        list_data_dict += read_data_file(os.path.join(sub_data_path, file))
                else:
                    list_data_dict += read_data_file(sub_data_path)
        else:
            list_data_dict = read_data_file(data_path)

        rank0_print(f"Dataset length: {len(list_data_dict)}.")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int) or isinstance(i, torch.Tensor) and i.numel() == 1:
            sources = [sources]

        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor

            try:
                if isinstance(image_folder, list):
                    for one_image_folder in image_folder:
                        if os.path.exists(os.path.join(one_image_folder, image_file)):
                            image = Image.open(os.path.join(one_image_folder, image_file)).convert('RGB')
                else:
                    image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                image  # if doesn't exit, it will raise error.
            except Exception as e:
                print(e, sources)
                return self.__getitem__(random.randint(0, len(self) - 1))

            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                image_size = image.size
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif self.data_args.image_aspect_ratio == "anyres":
                image_size = image.size
                image = process_anyres_image(  # torch.Size([5, 3, 336, 336])
                    image, processor, self.data_args.image_grid_pinpoints)
            elif self.data_args.image_aspect_ratio == "native_anyres":
                try:
                    inputs = processor(images=[image], return_tensors="pt")
                except Exception as e:
                    print(e, sources)
                    return self.__getitem__(random.randint(0, len(self) - 1))
                image = inputs['pixel_values']  # [ grid_H * grid_W, channel(1176) ]
                image_size = inputs['image_grid_thw'][0]  # [ 1, grid_H, grid_W ]
            else:
                image_size = image.size
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))

        if isinstance(i, int) or isinstance(i, torch.Tensor) and i.numel() == 1:
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['image_size'] = image_size
        elif self.data_args.is_multimodal:
            if self.data_args.image_aspect_ratio == "native_anyres":  # 448
                data_dict['image'] = torch.zeros(1024, 1176)  # [ num_token, channel ]
                data_dict['image_size'] = torch.tensor([1, 32, 32])  # [ t, image_h, image_wz ]
            else:
                # image does not exist in the data, but the model is multimodal
                crop_size = self.data_args.image_processor.crop_size
                data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
                data_dict['image_size'] = (crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    image_aspect_ratio: str

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            image_sizes = [instance['image_size'] for instance in instances]

            if self.image_aspect_ratio == "native_anyres":
                batch['images'] = torch.concat(images, dim=0)
                batch['image_sizes'] = torch.stack(image_sizes)
            else:
                if all(x is not None and x.shape == images[0].shape for x in images):
                    batch['images'] = torch.stack(images)
                else:
                    batch['images'] = images
                batch['image_sizes'] = image_sizes
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)

    SMOKE_TEST = bool(os.environ.get("SMOKE_TEST", 0))
    if SMOKE_TEST:
        dataset_len = 64
        train_dataset.list_data_dict = train_dataset.list_data_dict[:dataset_len]

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,
                                                     image_aspect_ratio=data_args.image_aspect_ratio)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
