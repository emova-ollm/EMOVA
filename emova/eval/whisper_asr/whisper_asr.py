import json
from pathlib import Path
from datasets import load_dataset, Dataset, Audio
from timeit import default_timer as timer
from tqdm import tqdm

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from .normalizers import EnglishTextNormalizer
import editdistance
from .CH_normalizer.cn_tn import TextNorm


# Todo: the input of whisper seems need to be 16kHz, while the audio currently is 22050Hz, needs resampling!

def load_whisper_model(model_path, device):
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=False, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        # batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    print('whisper model loaded!')
    return pipe



def recognize(wav_file_list, groundtruth_text_list, pipe, batch_size, print_verbose):
    t1 = timer()
    recognized_text_list = []
    for batch_start in tqdm(range(0, len(wav_file_list), batch_size)):
        batch_wav_file_list = wav_file_list[batch_start: min(batch_start + batch_size, len(wav_file_list))]
        batch_groundtruth_text_list = groundtruth_text_list[
                                      batch_start: min(batch_start + batch_size, len(wav_file_list))]
        batch_result_list = pipe(batch_wav_file_list)
        batch_recognized_text_list = [x['text'] for x in batch_result_list]
        recognized_text_list.extend(batch_recognized_text_list)
        if print_verbose:
            for groundtruth_text, recognized_text in zip(batch_groundtruth_text_list, batch_recognized_text_list):
                print(f'groundtruth text:{groundtruth_text}')
                print(f'recognized text:{recognized_text}')
    t2 = timer()
    time_passed = t2 - t1
    print(f"Computation Time: {time_passed:.4f} s")
    return recognized_text_list


def normalize_text(text_list):

    normalizer = TextNorm(
        to_banjiao=False,
        to_upper=False,
        to_lower=False,
        remove_fillers=False,
        remove_erhua=False,
        check_chars=False,
        remove_space=True,
        cc_mode='',  # currently seems 't2s' will have problem
    )

    normalized_text_list = []
    for original_text in tqdm(text_list):
        original_text = original_text.strip()
        normalized_text = normalizer(original_text)
        normalized_text_list.append(normalized_text)
    return normalized_text_list



def WER_calculation(groundtruth_text_list, recognized_text_list, normalized=False, print_verbose=False):
    error = 0
    word_count = 0
    for groundtruth_text, recognized_text in zip(groundtruth_text_list, recognized_text_list):
        if print_verbose:
            if normalized:
                print('normalized_groundtruth:' + groundtruth_text)
                print('normalized_recognized:' + recognized_text)
            else:
                print('groundtruth:' + groundtruth_text)
                print('recognized:' + recognized_text)
        ref_word_list = groundtruth_text.split()
        word_count += len(ref_word_list)
        rec_word_list = recognized_text.split()
        error += editdistance.eval(rec_word_list, ref_word_list)

    # print(f'model: {expr_name}; step: {step}')
    print(f"Word count: {word_count}")
    print(f"Word error: {error}")
    print(f'utterance num:{str(len(groundtruth_text_list))}')
    if normalized:
        print(f"WER with Whisper text normalization: {error / word_count:.4f} ")
    else:
        print(f"WER without Whisper text normalization: {error / word_count:.4f} ")

def EN_ASR_WER(pipe, wav_file_list, groundtruth_text_list, batch_size=24, print_verbose=False):

    recognized_text_list = recognize(wav_file_list, groundtruth_text_list, pipe, batch_size, print_verbose)

    # WER calculation without normalization
    WER_calculation(groundtruth_text_list, recognized_text_list, normalized=False,
                    print_verbose=print_verbose)

    # WER calculation with normalization
    normalizer = EnglishTextNormalizer()
    normalized_groundtruth_text_list, normalized_recognized_text_list = [], []
    for groundtruth_text, recognized_text in zip(groundtruth_text_list, recognized_text_list):
        normalized_groundtruth_text_list.append(normalizer(groundtruth_text))
        normalized_recognized_text_list.append(normalizer(recognized_text))
    WER_calculation(normalized_groundtruth_text_list, normalized_recognized_text_list, normalized=True,
                    print_verbose=print_verbose)

