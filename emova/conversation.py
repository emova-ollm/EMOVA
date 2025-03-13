import dataclasses
from enum import auto, Enum
from typing import List, Tuple
import base64
from io import BytesIO
from PIL import Image

import base64
tts_format = "Please synthesize the speech corresponding to the follwing text.\n"

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()
    GLM4 = auto()
    CHATML = auto()
    DeepSeekV2 = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.DeepSeekV2:
            seps = [self.sep, self.sep2]
            # if system_prompt == "" or system_prompt is None:
            #     ret = ""
            # else:
            if self.system:
                ret = self.system + seps[0]
            else:
                ret = ''
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if role == "User":
                        ret += "<｜sft▁begin｜>\n" + message + self.sep #<｜sft▁begin｜>User Input<｜sft▁end｜>\nResponse<｜end▁of▁sentence｜>
                    else:
                        ret += message + self.sep2
                else:
                    ret = ret
            return ret
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        elif self.sep_style == SeparatorStyle.GLM4:
            role = ("<|user|>", "<|assistant|>")
            ret = self.system + role[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += self.sep + message + role[(i + 1) % 2]
                else:
                    ret += ""
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if self.system == "" else self.system + self.sep + "\n"
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images, _ = message
                        message = "<image>" * len(images) + message
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        if isinstance(self.messages, tuple):
            self.messages += ([role, message],)
        else:
            self.messages.append([role, message])

    def process_image(self, image, image_process_mode, return_pil=False, image_format='PNG', max_len=1344, min_len=672):
        if image_process_mode == "Pad":
            def expand2square(pil_img, background_color=(122, 116, 104)):
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

            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
        if max(image.size) > max_len:
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
            longest_edge = int(shortest_edge * aspect_ratio)
            W, H = image.size
            if H > W:
                H, W = longest_edge, shortest_edge
            else:
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))
        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    image = self.process_image(image, image_process_mode, return_pil=return_pil)
                    images.append(image)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    img_b64_str = self.process_image(
                        image, "Default", return_pil=False,
                        image_format='JPEG')
                    img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


@dataclasses.dataclass
class Conversation_demo(Conversation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_prompt(self):
        """
        To make code clear, we only support LLaMA3, Qwen 2.5 and DeepSeekMoE series of LLMs.
        The other LLMs can be simply supported by comparing **Conversation_demo.get_prompt()** and **Conversation.get_prompt()**
        """
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple and messages[0][1][1] is not None:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message[:3]
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message[:3]
                    ret += role + message + self.sep
                else:
                    ret += role
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret
    
    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple and msg[1] is not None:
                    msg, image, image_process_mode = msg[:3]
                    image = self.process_image(image, image_process_mode, return_pil=return_pil)
                    images.append(image)
        return images
    
    def to_gradio_chatbot_public(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode, audio_input = msg
                    ret_msg = ""
                    if image is not None:
                        img_b64_str = self.process_image(
                            image, "Default", return_pil=False,
                            image_format='JPEG')
                        img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                        ret_msg += img_str
                    if audio_input is not None:
                        audio_b64_str = base64.b64encode(open(audio_input, "rb").read()).decode("utf-8")
                        audio_str = f'<audio src="data:audio/wav;base64,{audio_b64_str}" controls ></audio>'
                        ret_msg += audio_str
                    else:
                        ret_msg += msg.replace('<image>', '').replace(tts_format, '').strip()
                    ret.append([ret_msg, None])
                else:
                    ret.append([msg, None])
            else:
                if type(msg) is tuple:
                    audio_b64_str = base64.b64encode(open(msg[1], "rb").read()).decode("utf-8")
                    msg = f'<audio src="data:audio/wav;base64,{audio_b64_str}" controls autoplay></audio>'
                ret[-1][-1] = msg
        return ret
    
    def copy(self):
        return Conversation_demo(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)


conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
         "Renewable energy sources are those that can be replenished naturally in a relatively "
         "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
         "Non-renewable energy sources, on the other hand, are finite and will eventually be "
         "depleted, such as coal, oil, and natural gas. Here are some key differences between "
         "renewable and non-renewable energy sources:\n"
         "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
         "energy sources are finite and will eventually run out.\n"
         "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
         "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
         "and other negative effects.\n"
         "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
         "have lower operational costs than non-renewable sources.\n"
         "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
         "locations than non-renewable sources.\n"
         "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
         "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
         "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
         "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_plain = Conversation(
    system="",
    roles=("", ""),
    version="plain",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

conv_mistral_instruct = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="",
    sep2="</s>",
)

conv_chatml_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llama3 = Conversation(
    system="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.""",
    roles=("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    version="llama3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|eot_id|>",
)

conv_qwen2 = Conversation(
    system='<|im_start|>system\nYou are a helpful assistant.',
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="qwen2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>\n",
)

conv_glm4 = Conversation(
    system='[gMASK]<sop>',
    roles=("<|user|>\n", "<|assistant|>"),
    version="glm4",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.GLM4,
    sep="\n",
)

conv_phi3 = Conversation(
    system='<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.',
    roles=('<|user|>\n', '<|assistant|>\n'),
    version="phi3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|end|>\n",
)

conv_deepseek = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.",
    version="deepseek",
    roles=("<|User|>", "<|Assistant|>"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="\n\n",
    sep2="<｜end▁of▁sentence｜>",
    # stop_token_ids=[100001],
    # stop_str=["User:", "<｜end▁of▁sentence｜>"]
)

################
# For web demo
################
conv_vicuna_v1_demo = Conversation_demo(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1_demo",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_qwen2_demo = Conversation_demo(
    system='<|im_start|>system\nYou are a helpful assistant. Your name is emova, and you are purely developed by the emova Team.',
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="qwen2_demo",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>\n",
)

default_conversation = conv_vicuna_v1
default_conversation_demo = conv_vicuna_v1_demo
conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "vicuna_v1_demo": conv_vicuna_v1_demo,
    "llama_2": conv_llama_2,
    "mistral_instruct": conv_mistral_instruct,
    "chatml_direct": conv_chatml_direct,
    "mistral_direct": conv_chatml_direct,

    "plain": conv_plain,

    "mpt": conv_mpt,

    "llama3": conv_llama3,
    "qwen2": conv_qwen2,
    "qwen2_demo": conv_qwen2_demo,
    "glm4": conv_glm4,
    "phi3": conv_phi3,

    "deepseek": conv_deepseek,

}

if __name__ == "__main__":
    # print(default_conversation.get_prompt())
    print("deepseek template:")
    conv = conv_templates["deepseekv2"]
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi! This is Tony.")
    conv.append_message(conv.roles[0], "Who are you?")
    conv.append_message(conv.roles[1], "I am a helpful assistant.")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())