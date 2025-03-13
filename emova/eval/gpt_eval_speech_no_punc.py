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
    prompt_template = f'''Please rate the following response based on the criteria of helpfulness, relevance, accuracy, and comprehensiveness. Provide an overall score on a scale of 0 to 10, where a higher score indicates better overall performance.

- Helpfulness: How well does the response assist in addressing the question?
- Relevance: How closely does the response align with the question and the ground truth?
- Accuracy: How correct and factual is the response compared to the ground truth?
- Comprehensiveness: How thoroughly does the response cover the aspects of the question?

Please note that the evaluated response does not contain punctuation, but you should NOT give lower scores because of this, i.e., you should try to imagine there are punctuations or you could add them by youself.

Here is the question:
{input_item['gt_input']}

Here is the ground truth response for your reference:
{input_item['gt_output']}

Now, please evaluate the following response:
{input_item['pred_output']}

Provide your evaluation in JSON format as follows:
{{
    "reason": (str)  // Explanation of the score considering the criteria with no more than 100 words
    "score": (int),  // Overall score from 0 to 10
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
                time.sleep(random.randint(1, 30))
                print(f" Occurred: {e}. Retrying...")
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

output_score = []
fail_to_extract_data = []

for item in gpt_result:
    gpt_response_str = item["gpt4_response"]
    if gpt_response_str.startswith("```json"):
        gpt_response_str = gpt_response_str[7:].strip()
    if gpt_response_str.endswith("```"):
        gpt_response_str = gpt_response_str[:-3].strip()
    try:
        gpt_response_dict = json.loads(gpt_response_str)
        score = gpt_response_dict['score']
        output_score.append(int(score))
    except Exception as e:
        print(e)
        print(gpt_response_str)
        fail_to_extract_data.append(item)

print("Average score:", sum(output_score)/len(output_score))