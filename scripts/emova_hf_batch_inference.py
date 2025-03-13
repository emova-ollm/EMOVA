import torch
import json
import os
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import argparse

default_system_prompt = 'You are a helpful assistant.'


def parse_args():
    parser = argparse.ArgumentParser(description="Process images and questions with a pretrained model.")

    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Path to the pretrained model directory or name. For example: "/path/to/model"'
    )
    parser.add_argument(
        '--torch_dtype',
        type=str,
        default='fp32',
        help='data type for the modell'
    )

    parser.add_argument(
        '--system_prompt',
        type=str,
        default=None,
        help='System prompt to set the model\'s role and tone.'
    )

    parser.add_argument(
        '--json_path',
        type=str,
        default=None,
        required=True,
        help='Path to the JSON file containing input data. Should be a list, each element is a dict contain the question and the optional image .'
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default='output.json',
        help='Path to save the model responses as a JSON file.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Number of samples to process in each batch.'
    )

    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=1000,
        help='Maximum new tokens during inference.'
    )

    parser.add_argument(
        '--num_beams',
        type=int,
        default=1,
        help='Num beam in beam search.'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Maximum new tokens during inference.'
    )

    parser.add_argument(
        '--do_sample',
        action='store_true',
        help = 'do sampling during inference.',
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
    )

    parser.add_argument(
        '--device_map',
        type=str,
        default=None,
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Load data
    with open(args.json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"Read annotations from {args.json_path}")

    print(f"Load model from {args.model_name}, {args.device}")
    # Initialize processor, model, and tokenizer
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name,
                                      trust_remote_code=True,
                                      torch_dtype=torch.float16 if args.torch_dtype == 'fp16' else (torch.bfloat16 if args.torch_dtype=='bf16' else torch.float32),
                                      # attn_implementation=None,
                                      # attn_implementation='sdpa',
                                      attn_implementation='flash_attention_2',
                                      # attn_implementation='sdpa',
                                      # _attn_implementation='eager',
                                      device_map=args.device_map,
                                      device=args.device,
                                      )
    # model.to('cuda')
    print(model.config)
    print(f'model device {model.device} {args.device}')

    system_prompt = args.system_prompt if args.system_prompt else default_system_prompt
    batch_size = args.batch_size
    responses = []

    for batch_idx in range(0, len(data), batch_size):
        batch = data[batch_idx: batch_idx + batch_size]
        batch_images = []
        batch_text_prompts = []

        for item in batch:
            history_convs = []
            if 'history' in item:
                for conv in item['history']:
                    history_convs.extend([
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": conv['question']}],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": conv['response']}],
                        }
                    ])

            if 'image' in item:
                assert os.path.exists(item['image']), (item['image'])
                if len(history_convs):
                    history_convs[0]['content'].append({"type": "image"})

                    question_conv = [
                        {"type": "text", "text": item['question']},
                    ]
                else:
                    question_conv = [
                        {"type": "image"},
                        {"type": "text", "text": item['question']},
                    ]
                image = Image.open(item['image']).convert("RGB")
                batch_images.append(image)
            else:
                question_conv = [
                    {"type": "text", "text": item['question']},
                ]

            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                },
                *history_convs,
                {
                    "role": "user",
                    "content": question_conv
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, padding=True)
            batch_text_prompts.append(prompt)

        if len(batch_images) == 0:
            batch_images = None

        inputs = processor(
            batch_text_prompts,
            batch_images,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        if args.verbose:
            print(processor.tokenizer.batch_decode(inputs['input_ids']))

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens
            )

        all_response = []
        for input_ids, output_ids, item in zip(inputs['input_ids'], outputs, batch):
            # Extract the generated part
            response_ids = output_ids[len(input_ids):]
            response = processor.tokenizer.decode(response_ids, skip_special_tokens=True)

            if args.verbose:
                print(processor.tokenizer.decode(output_ids))
            item['response'] = response
            all_response.append(item)

        responses.extend(all_response)
        # break
    os.makedirs(os.path.split(args.output_path)[0], exist_ok=True)
    # Save responses to the output path
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)

    print(f"Responses have been saved to {args.output_path}")
