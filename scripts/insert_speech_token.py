import os
import logging
import argparse
import numpy as np

import tokenizers
from transformers import AutoTokenizer, AutoModelForCausalLM

def main(args):
    model_path = args.origin_model_path
    num_speech_tokens = args.num_speech_tokens
    saved_path = args.saved_model_path
    
    # load original tokenizer and model
    if 'deepseek' in model_path:
        from emova.model.language_model.deepseek_vl2 import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
        vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer
        model = DeepseekVLV2ForCausalLM.from_pretrained(model_path)
        model = model.language
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    original_embed_size, original_embed_dim = model.model.embed_tokens.weight.shape

    # show original vocab and embedding table
    print(f"original tokenizer.vocab_size: {tokenizer.vocab_size}\n")
    print(f"original len(tokenizer): {len(tokenizer)}\n")
    print(f"original embedding table shape:{model.model.embed_tokens.weight.shape}\n")
    print(f"original model: {model}\n")

    # add tokens
    speech_tokens = []
    for i in range(num_speech_tokens):
        speech_tokens.append(tokenizers.AddedToken(f'<|speech_{i}|>', single_word=True, rstrip=False, lstrip=False))

    # check and remove duplication
    # the new tokens should not include the special tokens like BOS
    new_tokens = set(speech_tokens) - set(tokenizer.vocab.keys())
    if new_tokens:
        added_tokens_num = tokenizer.add_tokens(list(new_tokens))
        new_tokenizer_size = len(tokenizer)
        if new_tokenizer_size > original_embed_size:
            new_embed_size = int(np.ceil(new_tokenizer_size / 128) * 128)
            model.resize_token_embeddings(new_embed_size)
        if added_tokens_num != num_speech_tokens:
            logging.warning(
                f"Actual numbers of added tokens {added_tokens_num} are different from proposed numbers {num_speech_tokens}: Probable duplication.")

    print(f"modified tokenizer.vocab_size: {tokenizer.vocab_size}\n")
    print(f"modified len(tokenizer): {len(tokenizer)}\n")
    print(f"modified embedding table shape:{model.model.embed_tokens.weight.shape}\n")
    print(f"modified model: {model}")

    # save modified model
    tokenizer.save_pretrained(saved_path)
    model.save_pretrained(saved_path)
    if 'deepseek' in model_path:
        vl_chat_processor.save_pretrained(saved_path)

    # sanity check
    text = "This is a speech: ###<|speech_0|><|speech_100|>"
    encoded_text = tokenizer.encode(text)
    print(f"Original text: {text}")
    print(f"Encoded text: {encoded_text}")
    for encoded_id in encoded_text:
        print(f"{encoded_id}: {tokenizer.decode(encoded_id)}")
    decoded_text = tokenizer.decode(encoded_text)
    print(f"Decoded text: {decoded_text}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--saved_model_path", type=str, default="./Qwen2.5-3B-Instruct_add_speech_token_4096_nostrip/")
    parser.add_argument("--num_speech_tokens", type=int, default=4096, 
                        help="This value should be consistent with the vocabulary size of the speech tokenizer. For EMOVA speech tokenizer, the vocabulary size is 4096 by default.")
    args = parser.parse_args()
    logging.info(f"args: {args}")
    
    main(args)