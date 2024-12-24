# 语镜VocaMirror——基于sensevoice、cosyvoice和qwen模型实现与“自身声音”对话

## 更新
v2    增加了预训练模式，允许使用预训练的声音作为对话声音源
      增加了噪声处理，提高抗环境干扰能力
## 项目简介
语镜 VocaMirror 系统受汤姆猫游戏和亲人语音克隆项目的启发，旨在实现用户与
“自己声音”对话的功能。该系统集成语音识别、自然语言处理和个性化语音合成技术，
应用了sensevoice、cosyvoice和qwen模型。具备以下应用价值：
- 趣味性互动：提供与自己声音对话的全新体验。
- 心理治疗辅助：熟悉的声音有助于缓解自闭症等患者的心理防御。
- 多功能拓展：可用作辩论训练、心灵自省等角色设定。

![界面.png](%E7%95%8C%E9%9D%A2.png)
<p style="text-align:center">（界面展示）</p>

VocaMirror 的主要功能包括：
- 语音识别 (ASR)：将用户语音转录为文本。
- 自然语言生成 (LLM)：基于上下文生成对话回复。
- 语音合成 (TTS)：将生成的文本转为个性化的语音回复。
- 用户界面：通过 Gradio 实现易用的前端界面。

## 环境安装与运行准备

``` bash
conda create -n cosyvoice python=3.8
conda activate cosyvoice
# pynini is required by WeTextProcessing, use conda to install it as it can be executed on all platform.
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# 您可以解压 ttsfrd 资源并安装 ttsfrd 包以获得更好的文本标准化性能
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd-0.3.6-cp38-cp38-linux_x86_64.whl

pip install gradio
pip install -r requirements.txt
pip install -U funasr modelscope
pip install dashscope
```

**前往github或魔搭社区下载cosyvoice模型与sensevoice模型,将```webui_qwen.py```放置在```cosyvoice/Cosyvoice```下**

cosyvoice魔搭链接：https://www.modelscope.cn/models/iic/CosyVoice-300M

sensevoice魔搭链接：https://www.modelscope.cn/models/iic/SenseVoiceSmall


## 代码解析
**1.引入必要的依赖**
```python
import gradio as gr
import os
import sys
import numpy as np
import torch
import argparse
import random
from dashscope import Generation
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav, logging
from funasr import AutoModel
import torchaudio
import librosa
import dashscope
```  

**2. 配置模型与参数**

2.1 DashScope API 配置
```python
dashscope.api_key = '申请的通义api_key'
```

2.2 初始化模型
```python
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
asr_model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 30000},
    trust_remote_code=True,
    disable_update=True,
)
prompt_sr, target_sr = 16000, 22050
```
CosyVoice：用于语音复刻的模型，支持从少量语音数据中生成相似的声音。

ASR 模型：用于将用户语音转录为文本。vad_model 提供语音活动检测以提升识别质量。

采样率设置：输入语音采样率为 16kHz，生成语音目标采样率为 22.05kHz。

**3. 语音处理**

3.1 音频后处理
```python
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > 0.8:
        speech = speech / speech.abs().max() * 0.8
    return torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
```

去除静音：通过 ```librosa.effects.trim``` 去除音频中的多余静音部分。

音量归一化：限制音频最大幅值为 ```0.8```，防止过大或失真。

3.2 ASR 推理
```python
def transcribe_audio(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    transcription = asr_model.generate(input=waveform[0].numpy())[0]["text"]
    transcription = re.sub(r"<\|.*?\|>", "", transcription).strip()
    return transcription

``` 

加载音频：使用 torchaudio 加载音频文件。

采样率调整：若音频采样率非 16kHz，则进行重采样。

ASR 推理：调用 funasr 模型进行语音转文本。

文本清洗：移除可能的冗余标记。

**4. 调用 DashScope 生成回复**

```python
def generate_reply(chat_query, chat_history):
    messages = [{'role': 'system', 'content': '你是一个友好的AI助手，请根据上下文作出回复。'}]
    for user_msg, ai_msg in chat_history:
        messages.append({'role': 'user', 'content': user_msg})
        messages.append({'role': 'assistant', 'content': ai_msg})
    messages.append({'role': 'user', 'content': chat_query})

    try:
        response = Generation.call(
            api_key=dashscope.api_key,
            model="qwen-plus",
            messages=messages,
            result_format="message"
        )
        if response.status_code == 200:
            reply_text = response.output.choices[0].message.content.strip()
        else:
            reply_text = f"出错了：HTTP返回码 {response.status_code}, 错误码 {response.code}, 错误信息 {response.message}"
    except Exception as e:
        reply_text = f"调用 API 发生异常：{str(e)}"

    chat_history.append((chat_query, reply_text))
    return reply_text, chat_history

```  
构建消息上下文：整合用户输入和聊天历史。

调用 DashScope：通过 Generation.call 生成回复。

异常处理：处理接口调用失败或异常情况。

**5. 语音合成**

```python
def generate_voice(input_audio, tts_text, prompt_text):
    prompt_speech_16k = postprocess(load_wav(input_audio, prompt_sr))
    for result in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False):
        yield (target_sr, result['tts_speech'].numpy().flatten())
```  

语音输入：通过加载输入音频的特征作为合成提示。

零样本生成：基于 CosyVoice 实现零样本语音合成。

**6. Gradio 界面设计**

语音复刻模式
```python
def main():
    with gr.Blocks() as demo:
        gr.Markdown("## VocaMirror - 不妨听听，自己的声音")
        with gr.Tabs() as tabs:
            with gr.Tab("语音复刻模式"):
                gr.Markdown("### 聊天框 - 语音输入 & 回复生成")
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbox = gr.Chatbot(label="对话历史")
                        input_audio = gr.Audio(label="输入音频", type="filepath")
                        submit_button = gr.Button("发送")

                    with gr.Column(scale=2):
                        output_audio = gr.Audio(label="AI 回复语音")

                state = gr.State([])

                def process_audio(audio_file, history):
                    text_input = transcribe_audio(audio_file)
                    reply_text, updated_history = generate_reply(text_input, history)
                    speech_generator = generate_voice(audio_file, reply_text, text_input)
                    output_audio_file = next(speech_generator, None)
                    return updated_history, updated_history, output_audio_file

                submit_button.click(
                    process_audio,
                    inputs=[input_audio, state],
                    outputs=[chatbox, state, output_audio]
                )

    demo.launch(share=False)
```  
``` webui_qwen.py```  为调用通义千问，``` webui_spark.py```  为调用星火大模型
