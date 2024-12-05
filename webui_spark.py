import gradio as gr
import os
import sys
import numpy as np
import torch
import argparse
import random
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav, logging
from sparkai.llm.llm import ChatSparkLLM
from funasr import AutoModel
import torchaudio
import librosa
from sparkai.core.messages import ChatMessage

# 配置星火大模型
SPARKAI_URL = 'xxxxxxxxxxxxxxxxxxxxxxxxxxx'
SPARKAI_APP_ID = 'xxxxxx'
SPARKAI_API_SECRET = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
SPARKAI_API_KEY = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
SPARKAI_DOMAIN = '4.0Ultra' # 模型可根据自身需求设置

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

# 初始化星火
spark = ChatSparkLLM(
    spark_api_url=SPARKAI_URL,
    spark_app_id=SPARKAI_APP_ID,
    spark_api_key=SPARKAI_API_KEY,
    spark_api_secret=SPARKAI_API_SECRET,
    spark_llm_domain=SPARKAI_DOMAIN,
    streaming=False,
)


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


# 调用星火生成回复
def generate_reply(chat_query, chat_history):
    # 构建消息上下文
    prompts = [ChatMessage(role='system', content="你是一个友好的AI助手，请根据上下文作出简洁回复，不超过50字。")]
    for user_msg, ai_msg in chat_history:
        prompts.append(ChatMessage(role='user', content=user_msg))
        prompts.append(ChatMessage(role='assistant', content=ai_msg))
    prompts.append(ChatMessage(role='user', content=chat_query))

    try:
        # 调用星火 LLM
        response = spark.generate([prompts])
        reply_text = response.generations[0][0].text.strip()
    except Exception as e:
        reply_text = f"出错了：{str(e)}"

    # 更新聊天记录
    chat_history.append((chat_query, reply_text))
    return reply_text, chat_history


# 语音合成
def generate_voice(input_audio, tts_text, prompt_text):  # audio原音频，回复文本
    prompt_speech_16k = postprocess(load_wav(input_audio, prompt_sr))
    for result in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False):
        yield (target_sr, result['tts_speech'].numpy().flatten())


# Gradio 界面设计
def main():
    with gr.Blocks() as demo:
        gr.Markdown("## VocaMirror - 不妨听听，自己的声音")

        with gr.Tabs() as tabs:
            # 页面 1: 原有功能
            with gr.Tab("语音复刻模式"):
                gr.Markdown("### SelfVoiceChat聊天框 - 语音输入 & 回复生成")

                with gr.Row():
                    with gr.Column(scale=3):
                        chatbox = gr.Chatbot(label="对话历史")
                        input_audio = gr.Audio(label="输入音频", type="filepath")
                        submit_button = gr.Button("发送")

                    with gr.Column(scale=2):
                        output_audio = gr.Audio(label="AI 回复语音")

                state = gr.State([])  # 保存聊天历史

                def process_audio(audio_file, history):
                    # 步骤 1: 转录语音
                    text_input = transcribe_audio(audio_file)
                    # 步骤 2: 获取 AI 回复
                    reply_text, updated_history = generate_reply(text_input, history)
                    print("reply text:", reply_text)
                    # 步骤 3: 合成语音
                    speech_generator = generate_voice(audio_file, reply_text, text_input)
                    output_audio_file = next(speech_generator, None)
                    return updated_history, updated_history, output_audio_file

                submit_button.click(
                    process_audio,
                    inputs=[input_audio, state],
                    outputs=[chatbox, state, output_audio]
                )

            # 页面 2: 预训练声音模式
            with gr.Tab("预训练声音模式"):
                gr.Markdown("### SelfVoiceChat聊天框 - 语音输入 & 回复生成 (预训练生成)")
                with gr.Row():
                    sft_dropdown = gr.Dropdown(choices=sft_spk, label='选择预训练音色', value=sft_spk[0], scale=0.25)
                    with gr.Column(scale=3):
                        chatbox_pretrained = gr.Chatbot(label="对话历史")
                        input_audio_pretrained = gr.Audio(label="输入音频", type="filepath")
                        submit_button_pretrained = gr.Button("发送")

                    with gr.Column(scale=2):
                        output_audio_pretrained = gr.Audio(label="AI 回复音频（预训练生成）")

                state_pretrained = gr.State([])  # 保存聊天历史

                def process_audio_pretrained(audio_file, history, selected_speaker):
                    # 步骤 1: 转录语音
                    text_input = transcribe_audio(audio_file)
                    # 步骤 2: 获取 AI 回复
                    reply_text, updated_history = generate_reply(text_input, history)
                    print("reply text (预训练):", reply_text)
                    # 步骤 3: 使用预训练音色生成语音
                    prompt_speech_16k = postprocess(load_wav(audio_file, prompt_sr))
                    speech_result = None

                    for result in cosyvoice.inference_sft(reply_text, selected_speaker, stream=False):
                        speech_result = result['tts_speech'].numpy().flatten()

                    if speech_result is not None:
                        return updated_history, updated_history, (target_sr, speech_result)
                    return updated_history, updated_history, None

                submit_button_pretrained.click(
                    process_audio_pretrained,
                    inputs=[input_audio_pretrained, state_pretrained, sft_dropdown],
                    outputs=[chatbox_pretrained, state_pretrained, output_audio_pretrained]
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
