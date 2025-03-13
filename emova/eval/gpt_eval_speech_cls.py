import json
import os
import argparse
import openai
from openai import OpenAI
import time
import random
import threading
from tqdm import tqdm

write_lock = threading.Lock()

def load_json(file_path):
    """
    Load a JSON file where each line is a separate JSON object.

    Args:
    file_path (str): The path to the JSON file.

    Returns:
    list: A list of dictionaries, each representing a JSON object.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]
    return data


def write_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')


def filter_prompts(in_file_path, out_file_path):
    in_data = load_json(in_file_path)
    temp_data = []
    for item in in_data:
        item["prompt"] = prepare_prompt(item)
        temp_data.append(item)
    return temp_data


def prepare_prompt(input_item):
    prompt_template = f'''To enhance the capabilities of multimodal large models in voice-based conversations, your task is to analyze the appropriate speech emotion and pitch for the assistant's response based on the text content of the user's question and the assistant's reply. Additionally, you need to score the assistant's response based on the actual situation.

Here is user's question:
{input_item['pred_input']}

Here is the assistant's response:
{input_item['pred_output']}

Here is the Assistant's Emotion Classification:
{input_item['pred_output_emotion']}

Here is the Assistant's Pitch Classification:
{input_item['pred_output_pitch']}


Please analyze the appropriate speech emotion and pitch that best match the assistant's response based on the text content of the user's question and the assistant's response.

**Emotion:**
First, analyze the assistant's response content and provide the speech emotion category and reason that you believe best matches the assistant's response in the voice conversation.  
The emotion options can only be selected from the following list: ['neutral', 'happy', 'sad', 'angry'].  
Then, analyze whether the "Assistant's Emotion Classification" is appropriate.  
If appropriate, the "Assistant's Emotion Classification Score" should be 1; otherwise, it should be 0.

**Pitch:**
First, analyze the assistant's response content and provide the speech pitch category and reason that you believe best matches the assistant's response in the voice conversation.  
The pitch options can only be selected from the following list: ['low', 'normal', 'high'].  
Then, analyze whether the "Assistant's Pitch Classification" is appropriate.  
If appropriate, the "Assistant's Pitch Classification Score" should be 1; otherwise, it should be 0.


Provide your evaluation in JSON format as follows:
{{
    "Assistant's Emotion Analysis": (str), // Analyze the response, propose emotion category and give the reason.
    "Assistant's Emotion Classification Score": (int),  // The score should be either 0 or 1, with 1 indicating appropriateness and 0 indicating inappropriateness.
    "Assistant's Pitch Analysis": (str), // Analyze the response, propose pitch category and give the reason.
    "Assistant's Pitch Classification Score": (int),  // The score should be either 0 or 1, with 1 indicating appropriateness and 0 indicating inappropriateness.
}}
Only output data in JSON format, no additional output required.
'''
    return prompt_template

def process_data(data_chunk, output_file_path, model, temperature, top_p):
    client = OpenAI(api_key='<KEY>')
    client.api_key = os.getenv("OPENAI_API_KEY", None)
    client.base_url = os.getenv("OPENAI_BASE_URL", 'https://api.openai.com/v1/')
    for row in data_chunk:
        msg = [
            {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
            {"role": "user", "content": row["prompt"]},
        ]
        success = False
        for attempt in range(trials):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=msg,
                    temperature=temperature,
                    top_p=top_p,
                )
            except Exception as e:
                print(e)
                time.sleep(random.randint(1, 30))
            except openai.error.APIerror:
                time.sleep(random.randint(1, 30))
            else:
                success = True
                break
        if success:
            response = response.choices[0].message.content

            row["gpt4_response"] = response
            row['gpt4_response_source'] = model
            line = json.dumps(row, ensure_ascii=False)

            with write_lock:
                with open(output_file_path, 'a+', encoding='utf-8') as f:
                    f.write(line + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, help="input file name with suffix")
    parser.add_argument("--model", type=str, default="gpt-4o", help="gpt-4-1106-preview, gpt-4-0613, ...")
    parser.add_argument("--temperature", type=float, default=0, help="temperature")
    parser.add_argument("--top_p", type=float, default=1, help="top_p")

    args = parser.parse_args()

    # Assuming 'trials' is defined somewhere in the code
    trials = 20

    input_file_path = args.input_file_path
    model = args.model
    temperature = args.temperature
    top_p = args.top_p

    input_file_name = input_file_path.split("/")[-1]
    input_file_dir = '/'.join(input_file_path.split("/")[:-1])
    output_file_name = os.path.splitext(input_file_name)[0] + '_{}_output_t{}_tp{}.json'.format(model, temperature, top_p)
    output_file_path = os.path.join(input_file_dir, output_file_name)

    input_data = filter_prompts(input_file_path, output_file_path)
    print(f"{len(input_data)} prompts.")

    num_threads = 30  # Define the number of threads

    # Split data into chunks for threading
    chunk_size = len(input_data) // num_threads
    data_chunks = [input_data[i:i+chunk_size] for i in range(0, len(input_data), chunk_size)]

    # progress_bar = tqdm(total=len(input_data), desc="Processing data")
    threads = []
    for data_chunk in data_chunks:
        thread = threading.Thread(target=process_data, args=(data_chunk, output_file_path, model, temperature, top_p))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("All threads completed.")


with open(output_file_path, 'r', encoding='utf-8') as file:
    gpt_result = [json.loads(line.strip()) for line in file]

emo_score_list = []
pitch_score_list = []
fail_to_extract_data = []

for item in gpt_result:
    gpt_response_str = item["gpt4_response"]
    if gpt_response_str.startswith("```json"):
        gpt_response_str = gpt_response_str[7:].strip()
    if gpt_response_str.endswith("```"):
        gpt_response_str = gpt_response_str[:-3].strip()
    try:
        gpt_response_dict = json.loads(gpt_response_str)


        emotion_analysis = gpt_response_dict.get("Assistant's Emotion Analysis", "")
        emotion_score = gpt_response_dict.get("Assistant's Emotion Classification Score", "")
        pitch_analysis = gpt_response_dict.get("Assistant's Pitch Analysis", "")
        pitch_score = gpt_response_dict.get("Assistant's Pitch Classification Score", "")

        if emotion_analysis and pitch_analysis and isinstance(emotion_score, int) and isinstance(pitch_score, int):
            emo_score_list.append(int(emotion_score))
            pitch_score_list.append(int(pitch_score))

    except Exception as e:
        print(e)
        print(gpt_response_str)
        fail_to_extract_data.append(item)

print("Average emo score:", sum(emo_score_list)/len(emo_score_list))
print("Average pitch score:", sum(pitch_score_list)/len(pitch_score_list))