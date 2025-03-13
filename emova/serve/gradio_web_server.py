import argparse
import datetime
import json
import os
import uuid
import time
import hashlib
import shutil

import gradio as gr
import requests

import torch
from transformers import AutoModel

from emova.conversation import default_conversation_demo, conv_templates, SeparatorStyle
from emova.constants import LOGDIR
from emova.utils import build_logger, server_error_msg, violates_moderation, moderation_msg

try:
    import torch_npu
    from torch_npu.npu import amp
    from torch_npu.contrib import transfer_to_npu
    print('Successful import torch_npu')
except Exception as e:
    print(e)

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "EMOVA Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

##########################################
# Audio part
##########################################
speech_tokenizer = AutoModel.from_pretrained("Emova-ollm/emova_speech_tokenizer_hf", torch_dtype=torch.float32, trust_remote_code=True).eval()
speech_tokenizer = speech_tokenizer.cuda()

####################
# task format
####################
asr_format = "Please recognize the text corresponding to the follwing speech.\n"
tts_format = "Please synthesize the speech corresponding to the follwing text.\n"
chat_format = r'Please recognize the texts, emotion and pitch from the user question speech units and provide the texts, emotion, pitch and speech units for the assistant response. \nEmotion should be chosen from ["neutral", "happy", "sad", "angry", "surprised", "disgusted", "fearful"]. \nPitch should be chosen from ["low", "normal", "high"].\nYour output should be in json format.\nAn output example is:\n{"user question text": "", "user question emotion": "", "user question pitch": "", "assistant response text": "", "assistant response emotion": "", "assistant response pitch": ""，"assistant response speech": ""}\n\nuser question speech:'

def s2u_asr(text, audio_file):
    return asr_format + speech_tokenizer.encode(audio_file)

def s2u_chat(text, audio_file):
    return chat_format + speech_tokenizer.encode(audio_file)

def u2s_tts(text, audio_file):
    return tts_format + text

mode2func = dict(
    asr=s2u_asr,
    chat=s2u_chat,
    tts=u2s_tts,
)

##########################################
# Gradio Loading part
##########################################
def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort()
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(value=model, visible=True)

    state = default_conversation_demo.copy()

    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation_demo.copy()
    dropdown_update = gr.Dropdown(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update


##########################################
# Gradio Generate part
##########################################
def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode, *prev_human_msg[1][3:])
    state.skip_next = False
    return (state, state.to_gradio_chatbot_public(), "", None, None) + (disable_btn,) * 2


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation_demo.copy()
    return (state, state.to_gradio_chatbot_public(), "", None) + (disable_btn,) * 2 + (None,)


############
# Show prompt in the chatbot
# Input: [state, textbox, imagebox, image_process_mode, audio_input, audio_mode], request
# Return: [state, chatbot, textbox, imagebox, audio_input] + btn_list                 
############
def add_text(state, text, image, image_process_mode, audio_input, audio_mode, request: gr.Request):
    ############
    # Input legality checking
    ############
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None and audio_input is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot_public(), "", None, None) + (no_change_btn,) * 2
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot_public(), moderation_msg, None, None) + (no_change_btn,) * 2
    
    ############
    # Re-initialize if having conducted audio conversations
    ############
    for i, (role, msg) in enumerate(state.messages[state.offset:]):
        if isinstance(msg, tuple) and msg[-1] is not None:
            state = default_conversation_demo.copy()
            break
    
    ############
    # Deal with image inputs
    ############
    if image is not None:
        if '<image>' not in text:
            text = text + '\n<image>'
        text = (text, image, image_process_mode, None)
        state = default_conversation_demo.copy()
    
    ############
    # Deal with audio inputs
    ############
    if audio_input is not None or audio_mode == 'tts':
        if isinstance(text, tuple):
            if audio_mode == 'chat':
                prompt = mode2func[audio_mode](text[0][:-len("\n<image>")], audio_input)
                text = (prompt + "\n<image>", text[1], text[2], audio_input)
            elif audio_mode == 'tts':
                prompt = mode2func[audio_mode](text[0][:-len("\n<image>")], audio_input)
                text = (prompt, None, None, None)
            else:
                prompt = mode2func[audio_mode](text, audio_input)
                text = (prompt, None, None, audio_input)
        else:
            prompt = mode2func[audio_mode](text, audio_input)
            text = (prompt, None, None, audio_input)
        state = default_conversation_demo.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    logger.info(str(state.messages))
    return (state, state.to_gradio_chatbot_public(), "", None, None) + (disable_btn,) * 2


############
# get response
# Input: [state, model_selector, temperature, top_p, max_output_tokens, speaker], request
# Return: [state, chatbot] + btn_list,
############
def http_bot(state, model_selector, temperature, top_p, max_new_tokens, speaker, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot_public()) + (no_change_btn,) * 2
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if "qwen2" in model_name.lower() or 'qwen-2' in model_name.lower():
            template_name = 'qwen2_demo'
        else:
            template_name = "vicuna_v1"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
                        json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot_public(), enable_btn, enable_btn)
        return

    # Construct prompt
    prompt = state.get_prompt()

    # save serve images and audios
    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    if type(state.messages[-2][-1]) is tuple and state.messages[-2][-1][-1] is not None:
        tmp_audio_path = state.messages[-2][-1][-1]
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_audios", f"{t.year}-{t.month:02d}-{t.day:02d}", tmp_audio_path.split("/")[-2] + '.wav')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        shutil.copyfile(tmp_audio_path, filename)
    
    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": int(max_new_tokens),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": f'List of {len(state.get_images())} images: {all_image_hash}',
    }
    logger.info(f"==== request ====\n{pload}")

    pload['images'] = state.get_images()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot_public()) + (disable_btn,) * 2

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
                                 headers=headers, json=pload, stream=True, timeout=10)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    if tts_format not in prompt and chat_format not in prompt:
                        state.messages[-1][-1] = output + "▌"
                    else:
                        state.messages[-1][-1] = "▌"
                    yield (state, state.to_gradio_chatbot_public()) + (disable_btn,) * 2
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot_public()) + (enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot_public()) + (enable_btn, enable_btn)
        return

    ################
    # decode output to audio
    ################
    temp_file = None
    if tts_format in prompt or chat_format in prompt:
        try:
            try:
                if output.startswith("{"):
                    if output.endswith("|>"):
                        output += "\"}"
                    elif output.endswith("\""):
                        output += "}"
                info_dict = json.loads(output)
                content_unit = info_dict['assistant response speech'].replace('<|speech_', '').replace('|>', ' ').strip()
                emotion = info_dict['assistant response emotion'] if 'assistant response emotion' in info_dict else "neutral"
                speed = info_dict['assistant response speed'] if 'assistant response speed' in info_dict else "normal"
                pitch = info_dict['assistant response pitch'] if 'assistant response pitch' in info_dict else "normal"
                gender = speaker.lower() if speaker else 'female'
            except:
                content_unit = output.replace('<|speech_', '').replace('|>', ' ').strip()
                emotion = 'neutral'
                speed = "normal"
                pitch = "normal"
                gender = speaker.lower() if speaker else 'female'
            
            condition = f'gender-{gender}_emotion-{emotion}_speed-{speed}_pitch-{pitch}'
            print(condition)
        
            id = str(uuid.uuid4())
            os.makedirs("./demo_audio", exist_ok=True)    
            speech_tokenizer.decode(content_unit, condition=condition, output_wav_file=f"./demo_audio/{id}_temp_audio.wav")
            temp_file = f"./demo_audio/{id}_temp_audio.wav"
        except Exception as e:
            print(e)

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    if tts_format in prompt or chat_format in prompt:
        if temp_file is not None:
            state.messages[-1][-1] = (output, temp_file)
            yield (state, state.to_gradio_chatbot_public()) + (enable_btn,) * 2
        else:
            state.messages[-1][-1] = server_error_msg
            yield (state, state.to_gradio_chatbot_public()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
    else:
        yield (state, state.to_gradio_chatbot_public()) + (enable_btn,) * 2

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

title_markdown = ("""
<div style="display: flex; align-items: center; padding: 20px; border-radius: 10px; background-color: #f0f0f0;">
  <div style="margin-left: 20px; margin-right: 40px;">
    <img src="https://emova-ollm.github.io/static/images/icons/emova.png" alt="Icon" style="width: 100px; height: 100px; border-radius: 10px;">
  </div>
  <div>
    <h1 style="margin: 0;">EMOVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions</h1>
    <h2 style="margin: 10px 0;">📃 <a href="https://arxiv.org/abs/2409.18042" style="font-weight: 300;">Paper</a> | 💻 <a href="https://github.com/emova-ollm/EMOVA" style="font-weight: 300;">Code</a> | 🤗 <a href="https://huggingface.co/Emova-ollm" style="font-weight: 300;">HuggingFace</a> | 🌐 <a href="https://emova-ollm.github.io/" style="font-weight: 300;">Website</a></h2>
    <p  style="margin: 20px 0;">
      <strong>1. Note that to use the Webcam and Microphone, open <a href="chrome://flags/#unsafely-treat-insecure-origin-as-secure">chrome://flags/#unsafely-treat-insecure-origin-as-secure</a> and put this link into the box.</strong><br/>
      <strong>2. To chat with EMOVA, upload images, enter texts or record audios and then do not forget to <mark>Click 💬 Chat Button</mark> ^v^!</strong><br/>
      <strong>3. Heighten the <code>Max output tokens</code> if necessary to talk longer with EMOVA.</strong>
    </p>
  </div>
</div>
""")

tos_markdown = ("""
## Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
""")

learn_more_markdown = ("""
## License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/QwenLM/Qwen/blob/main/LICENSE) of Qwen and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.

## Acknowledgement
The service is built upon [LLaVA](https://github.com/haotian-liu/LLaVA/). We thanks the authors for open-sourcing the wonderful code.

## Citation
<pre><code>@article{chen2024emova,
  title={Emova: Empowering language models to see, hear and speak with vivid emotions},
  author={Chen, Kai and Gou, Yunhao and Huang, Runhui and Liu, Zhili and Tan, Daxin and Xu, Jing and Wang, Chunwei and Zhu, Yi and Zeng, Yihan and Yang, Kuo and others},
  journal={arXiv preprint arXiv:2409.18042},
  year={2024}
}</code></pre>
""")

block_css = """
#buttons button {
    min-width: min(120px,100%);
}

.message-row img {
    margin: 0px !important;
}

.avatar-container img {
    padding: 0px !important;
}
"""


############
# Layout Demo
############
def build_demo(embed_mode, cur_dir=None, concurrency_count=10):
    textbox = gr.Textbox(label="Text", show_label=False, placeholder="Enter text or record audio in the right and then click 💬 Chat to talk with me ^v^", container=False, scale=6)
    audio_input = gr.Audio(label="Audio", sources=["microphone", "upload"], type="filepath", max_length=10, show_download_button=True, waveform_options=dict(sample_rate=16000), scale=2)
    with gr.Blocks(title="EMOVA", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        ##############
        # Chatbot
        ##############
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)

                imagebox = gr.Image(type="pil", label="Image")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image", visible=False)

                ##############
                # Parameters
                ##############
                with gr.Accordion("Parameters", open=True) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature")
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P")
                    max_output_tokens = gr.Slider(minimum=0, maximum=2048, value=1024, step=32, interactive=True, label="Max output tokens")
                    speaker = gr.Radio(["Female", "Male"], value="Female", label="Speaker")

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="EMOVA Chatbot",
                    height=460, # TODO
                    layout="bubble",
                    avatar_images=["./emova/serve/examples/user_avator.png", "./emova/serve/examples/icon_256.png"]
                )
                with gr.Row(equal_height=True):
                    textbox.render()
                    audio_input.render()
                with gr.Row(elem_id="buttons") as button_row:
                    submit_btn = gr.Button(value="💬  Chat", variant="primary")
                    audio_asr_btn = gr.Button(value="🎧  ASR", interactive=True, visible=False)
                    audio_tts_btn = gr.Button(value="🔊  TTS", interactive=True, visible=False)
                    #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)
        
        ##############
        # Examples
        ##############
        if cur_dir is None:
            cur_dir = os.path.dirname(os.path.abspath(__file__))

        with gr.Row():
            with gr.Column(scale=9):
                gr.Examples(examples=[
                    [f"{cur_dir}/examples/emo-speech/what_is_your_name.wav"],
                    [f"{cur_dir}/examples/emo-speech/I_am_so_sad.wav"],
                    [f"{cur_dir}/examples/emo-speech/parent.wav"],
                    [f"{cur_dir}/examples/emo-speech/wedding(CH).wav"],
                ], inputs=[audio_input], label='Audio Examples (Click to load the examples~)')

        with gr.Row(equal_height=True):
            gr.Examples(examples=[
                [f"{cur_dir}/examples/image-text/example_1.png", "Why is this image funny?"],
                [f"{cur_dir}/examples/image-text/example_2.png", "First please perform reasoning, and think step by step to provide best answer to the following question:\n\nWhat is the original price for pork belly before discount?"],
                [f"{cur_dir}/examples/image-text/example_3.png", "Convert this table to markdown format."],
            ], inputs=[imagebox, textbox], label='Image Examples')
            gr.Examples(examples=[
                [f"{cur_dir}/examples/emo-speech/write_a_poem.jfif", f"{cur_dir}/examples/emo-speech/write_a_poem.wav"],
                [f"{cur_dir}/examples/emo-speech/I_am_happy_get_my_offer.png", f"{cur_dir}/examples/emo-speech/I_am_happy_get_my_offer.wav"],
                [f"{cur_dir}/examples/structure-speech/names_of_main_actors.jpg", f"{cur_dir}/examples/structure-speech/names_of_main_actors.wav"],
            ], inputs=[imagebox, audio_input], label='Omni Examples 1')
            gr.Examples(examples=[
                [f"{cur_dir}/examples/structure-speech/how_to_save_water.png", f"{cur_dir}/examples/structure-speech/how_to_save_water.wav"],
                [f"{cur_dir}/examples/structure-speech/internet_coverage.png", f"{cur_dir}/examples/structure-speech/internet_coverage.wav"],
                [f"{cur_dir}/examples/structure-speech/how_to_use_website.PNG", f"{cur_dir}/examples/structure-speech/how_to_use_website.wav"],
            ], inputs=[imagebox, audio_input], label='Omni Examples 2')

        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [regenerate_btn, clear_btn]
        regenerate_btn.click(
            regenerate,
            [state, image_process_mode],
            [state, chatbot, textbox, imagebox, audio_input] + btn_list
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, speaker],
            [state, chatbot] + btn_list,
            # concurrency_limit=concurrency_count
        )

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, imagebox] + btn_list + [audio_input],
            queue=False
        )

        textbox.submit(
            add_text,
            [state, textbox, imagebox, image_process_mode, audio_input, gr.Number(value='chat', visible=False)],
            [state, chatbot, textbox, imagebox, audio_input] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, speaker],
            [state, chatbot] + btn_list,
            # concurrency_limit=concurrency_count
        )

        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode, audio_input, gr.Number(value='chat', visible=False)],
            [state, chatbot, textbox, imagebox, audio_input] + btn_list
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, speaker],
            [state, chatbot] + btn_list,
            # concurrency_limit=concurrency_count
        )
        
        ##############
        # Audio buttons
        ##############
        audio_asr_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode, audio_input, gr.Number(value='asr', visible=False)],
            [state, chatbot, textbox, imagebox, audio_input] + btn_list
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, speaker],
            [state, chatbot] + btn_list,
            # concurrency_limit=concurrency_count
        )
        audio_tts_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode, audio_input, gr.Number(value='tts', visible=False)],
            [state, chatbot, textbox, imagebox, audio_input] + btn_list
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, speaker],
            [state, chatbot] + btn_list,
            # concurrency_limit=concurrency_count
        )

        ##############
        # Demo loading
        ##############
        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                js=get_window_url_params
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=16)
    parser.add_argument("--model-list-mode", type=str, default="once",
                        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    demo = build_demo(args.embed, concurrency_count=args.concurrency_count)
    demo.queue(
        max_size=10,
        api_open=False
    ).launch(
        favicon_path="./emova/serve/examples/icon_256.png",
        allowed_paths=["/"],
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
