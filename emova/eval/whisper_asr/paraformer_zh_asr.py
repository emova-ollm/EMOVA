from pathlib import Path
from timeit import default_timer as timer
from tqdm import tqdm
import editdistance

from funasr import AutoModel

from .CH_normalizer.cn_tn import TextNorm


def load_paraformer_zh_model(model_path, device):
    model = AutoModel(model=model_path, model_revision="v2.0.4",
                      # vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                      # punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                      # spk_model="cam++", spk_model_revision="v2.0.2",
                      device=device,
                      )
    return model


def recognize(wav_file_list, groundtruth_text_list, pipe, batch_size, print_verbose):
    t1 = timer()
    recognized_text_list = []
    for batch_start in tqdm(range(0, len(wav_file_list), batch_size)):
        batch_wav_file_list = wav_file_list[batch_start: min(batch_start + batch_size, len(wav_file_list))]
        # print('\n'.join(batch_wav_file_list))
        batch_groundtruth_text_list = groundtruth_text_list[
                                      batch_start: min(batch_start + batch_size, len(wav_file_list))]
        batch_result_list = pipe.generate(batch_wav_file_list, batch_size=batch_size)
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
        if normalized_text != original_text:
            # print(f"original_text {original_text} is different from \nnormalized_text {normalized_text}")
            pass
        normalized_text_list.append(normalized_text)

    return normalized_text_list


def write_output(output_file, wav_file_list, recognized_text_list, groundtruth_text_list, normalized):
    with open(output_file, 'w', encoding='utf-8') as f:
        for (wav_file, recognized_text, groundtruth_text) in zip(wav_file_list, recognized_text_list,
                                                                 groundtruth_text_list):
            f.write(f"wav_file: {wav_file}\n")
            if not normalized:
                f.write(f"recognized_text: {recognized_text}\n")
                f.write(f"groundtruth_text: {groundtruth_text}\n\n")
            else:
                f.write(f"normalized_recognized_text: {recognized_text}\n")
                f.write(f"normalized_groundtruth_text: {groundtruth_text}\n\n")


def CER_calculation(groundtruth_text_list, recognized_text_list, normalized=False, print_verbose=False):
    error = 0
    character_count = 0
    error_list = []
    for groundtruth_text, recognized_text in zip(groundtruth_text_list, recognized_text_list):
        if print_verbose:
            if normalized:
                print('normalized_groundtruth:' + groundtruth_text)
                print('normalized_recognized:' + recognized_text)
            else:
                print('groundtruth:' + groundtruth_text)
                print('recognized:' + recognized_text)
        ref_character_list = list(groundtruth_text)
        character_count += len(ref_character_list)
        rec_character_list = list(recognized_text)
        ee = editdistance.eval(rec_character_list, ref_character_list)
        error += ee
        error_list.append(ee)
    
    sort_index = sorted(enumerate(error_list), key=lambda x: x[1])[::-1]
    # for idx, _ in sort_index:
    #     print('rank groundtruth:' + groundtruth_text_list[idx])
    #     print('rank recognized:' + recognized_text_list[idx])
    # print(f'model: {expr_name}; step: {step}')
    print(f"Word count: {character_count}")
    print(f"Word error: {error}")
    print(f'utterance num: {str(len(groundtruth_text_list))}')
    if normalized:
        print(f"CER with text normalization: {error / character_count:.4f} ")
    else:
        print(f"CER without text normalization: {error / character_count:.4f} ")

    


def CH_ASR_CER(pipe, wav_file_list, groundtruth_text_list, batch_size=24, print_verbose=False):

    recognized_text_list = recognize(wav_file_list, groundtruth_text_list, pipe, batch_size, print_verbose)

    # CER calculation without normalization
    CER_calculation(groundtruth_text_list, recognized_text_list, normalized=False,
                    print_verbose=print_verbose)

    # CER calculation with normalization
    normalized_recognized_text_list = normalize_text(recognized_text_list)
    normalized_groundtruth_text_list = normalize_text(groundtruth_text_list)
    CER_calculation(normalized_groundtruth_text_list, normalized_recognized_text_list, normalized=True,
                    print_verbose=print_verbose)
