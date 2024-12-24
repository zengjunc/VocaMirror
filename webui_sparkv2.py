import os
import sys
import tempfile
import argparse
import torch
import torchaudio
import librosa
import soundfile as sf
import re
import gradio as gr
from torchaudio.transforms import Resample
from denoiser import pretrained
from denoiser.dsp import convert_audio
from sparkai.llm.llm import ChatSparkLLM
from sparkai.core.messages import ChatMessage
from funasr import AutoModel
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

# 配置星火大模型
SPARKAI_URL = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
SPARKAI_APP_ID = 'XXXXXXXX'
SPARKAI_API_SECRET = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
SPARKAI_API_KEY = 'XXXXXXXXXXXXXXXXXXXXXXXXXXX'
SPARKAI_DOMAIN = 'XXXXXXX'

# 初始化星火对话模型
spark = ChatSparkLLM(
    spark_api_url=SPARKAI_URL,
    spark_app_id=SPARKAI_APP_ID,
    spark_api_key=SPARKAI_API_KEY,
    spark_api_secret=SPARKAI_API_SECRET,
    spark_llm_domain=SPARKAI_DOMAIN,
    streaming=False,
)

# 初始化 ASR 模型
asr_model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 30000},
    trust_remote_code=True,
    disable_update=True,
)

# 初始化降噪模型
denoiser_model = pretrained.dns64().cpu()

# 初始化 CosyVoice 模型
cosyvoice = None
prompt_sr, target_sr = 16000, 22050

# 音频降噪
def denoise_audio(input_audio_path, output_audio_path):
    """
    使用 denoiser 模型对音频降噪
    :param input_audio_path: 输入音频路径
    :param output_audio_path: 输出音频路径
    """
    wav, sr = torchaudio.load(input_audio_path)
    resampler = Resample(orig_freq=sr, new_freq=16000)
    wav_resampled = resampler(wav)
    wav_resampled = convert_audio(wav_resampled.cpu(), 16000, denoiser_model.sample_rate, denoiser_model.chin)
    with torch.no_grad():
        denoised = denoiser_model(wav_resampled[None])[0]
    sf.write(output_audio_path, denoised.cpu().numpy().T, 16000)

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

# ASR 转录
def transcribe_audio(audio_path):
    """
    对音频进行转录
    :param audio_path: 输入音频路径
    :return: 转录文本
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    transcription = asr_model.generate(input=waveform[0].numpy())[0]["text"]

    # 移除可能的标记（语言、心情等）
    transcription = re.sub(r"<\|.*?\|>", "", transcription).strip()
    return transcription

# 获取 AI 回复
def generate_reply(chat_query, chat_history):
    """
    获取 AI 对话生成的回复
    :param chat_query: 用户输入的文本
    :param chat_history: 对话历史
    :return: 回复文本和更新后的对话历史
    """
    prompts = [ChatMessage(role='system', content="你是一个友好的AI助手，请根据上下文作出简洁回复，不超过50字。")]
    for user_msg, ai_msg in chat_history:
        prompts.append(ChatMessage(role='user', content=user_msg))
        prompts.append(ChatMessage(role='assistant', content=ai_msg))
    prompts.append(ChatMessage(role='user', content=chat_query))

    try:
        response = spark.generate([prompts])
        reply_text = response.generations[0][0].text.strip()
    except Exception as e:
        reply_text = f"出错了：{str(e)}"

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
        gr.Markdown("## VocaMirror - 听听自己的声音")

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
                    # 动态生成降噪输出路径
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        denoised_audio = temp_file.name

                    # 步骤 1: 噪声处理
                    denoise_audio(audio_file, denoised_audio)

                    # 步骤 2: 转录语音
                    text_input = transcribe_audio(denoised_audio)

                    # 步骤 3: 获取 AI 回复
                    reply_text, updated_history = generate_reply(text_input, history)

                    # 步骤 4: 合成语音
                    speech_generator = generate_voice(audio_file, reply_text, text_input)
                    output_audio_file = next(speech_generator, None)

                    return updated_history, updated_history, denoised_audio, output_audio_file

                submit_button.click(
                    process_audio,
                    inputs=[input_audio, state],
                    outputs=[chatbox, state, gr.Audio(label="降噪后音频"), output_audio]
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

                    # 步骤 3: 使用预训练音色生成语音
                    prompt_speech_16k = postprocess(load_wav(audio_file, prompt_sr))
                    speech_result = None

                    for result in cosyvoice.inference_sft(reply_text, selected_speaker, stream=False):
                        speech_result = result['tts_speech'].numpy().flatten()

                    if speech_result is not None:
                        return updated_history, updated_history, audio_file, (target_sr, speech_result)
                    return updated_history, updated_history, audio_file, None

                submit_button_pretrained.click(
                    process_audio_pretrained,
                    inputs=[input_audio_pretrained, state_pretrained, sft_dropdown],
                    outputs=[chatbox_pretrained, state_pretrained, gr.Audio(label="降噪后音频"),
                             output_audio_pretrained]
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
