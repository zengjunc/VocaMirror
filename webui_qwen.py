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

dashscope.api_key = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
# 配置 CosyVoice 和 ASR 模型
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
asr_model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 30000},
    trust_remote_code=True,
    disable_update=True,
)
prompt_sr, target_sr = 16000, 22050

# 音频后处理
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > 0.8:
        speech = speech / speech.abs().max() * 0.8
    return torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)

# ASR 模型推理
import re
def transcribe_audio(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    transcription = asr_model.generate(input=waveform[0].numpy())[0]["text"]

    # 移除可能的标记（语言、心情等）
    transcription = re.sub(r"<\|.*?\|>", "", transcription).strip()
    return transcription

# 调用 DashScope API 生成回复
def generate_reply(chat_query, chat_history):
    # 构建消息上下文
    messages = [{'role': 'system', 'content': '你是一个友好的AI助手，请根据上下文作出回复。'}]
    # messages = [{'role': 'system', 'content': '请扮演我的思考伙伴，帮助我进行更加严谨和深入的思考。在对话过程中，请根据我提供的上下文信息给出反馈或提出问题，以促进更全面的分析与理解。'}]
    for user_msg, ai_msg in chat_history:
        messages.append({'role': 'user', 'content': user_msg})
        messages.append({'role': 'assistant', 'content': ai_msg})
    messages.append({'role': 'user', 'content': chat_query})

    try:
        # 调用 DashScope 的 Qwen-Plus 模型
        response = Generation.call(
            api_key=os.getenv("sk-9180721f147e4c8da8592600ff865ee1"),  # 或者直接使用字符串替换 os.getenv("DASHSCOPE_API_KEY")
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

    # 更新聊天记录
    chat_history.append((chat_query, reply_text))
    return reply_text, chat_history

# 语音合成
def generate_voice(input_audio, tts_text, prompt_text):
    prompt_speech_16k = postprocess(load_wav(input_audio, prompt_sr))
    for result in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False):
        yield (target_sr, result['tts_speech'].numpy().flatten())

# Gradio 界面设计
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

                state = gr.State([])  # 保存聊天历史

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice-300M-Instruct',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir)
    sft_spk = cosyvoice.list_avaliable_spks()
    main()
