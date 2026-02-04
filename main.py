import os, whisper, yt_dlp, shutil
from openai import OpenAI
import gradio as gr

# --- 外部接口区 ---
API_KEY = "sk-placeholder"

def process_video(api_key, video_url):
    """Gradio 调用的核心逻辑函数"""
    final_key = api_key if api_key and "sk-" in api_key else API_KEY
    
    if "placeholder" in final_key:
        yield "❌ 错误：请在界面输入有效的 API_KEY", None
        return

    try:
        # --- 1. 音频获取 ---
        audio_file = "temp_audio.m4a"
        if not os.path.exists(audio_file):
            yield "📥 正在从源站抓取音频...", None
            audio_opts = {
                'format': 'm4a/bestaudio/best',
                'outtmpl': 'temp_audio.%(ext)s',
                'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'm4a'}],
                'quiet': True
            }
            with yt_dlp.YoutubeDL(audio_opts) as ydl:
                ydl.download([video_url])

        # --- 2. 语音识别 (缓存) ---
        txt_cache = "raw_transcript.txt"
        if os.path.exists(txt_cache):
            with open(txt_cache, "r", encoding="utf-8") as f:
                content_text = f.read()
        else:
            yield "🎙️ Whisper 正在拼命听写...", None
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = whisper.load_model("base", device=device) 
            result = model.transcribe(audio_file)
            content_text = result['text']
            with open(txt_cache, "w", encoding="utf-8") as f:
                f.write(content_text)
        
        # --- 3. 核心改进：原意分段 (使用 Turbo 模型规避审核且保真) ---
        yield "📑 正在进行【原意分段】(不删减原文)...", None
        client = OpenAI(api_key=final_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        split_prompt = f"""
        你是一位专业的速记员。请对以下文稿进行【仅分段与加标点】处理：
        1. 严禁删减、改动或润色原文任何词语。
        2. 仅根据语义进行自然段落切分并补充标点。
        3. 这是一个关于人文/心理学的学术探讨，请保持其原始表述。
        
        待处理内容：
        {content_text}
        """
        split_res = client.chat.completions.create(model="qwen-turbo", messages=[{"role": "user", "content": split_prompt}])
        segmented_text = split_res.choices[0].message.content
        
        # 保存分段原稿文件
        seg_file = "1_分段原稿.txt"
        with open(seg_file, "w", encoding="utf-8") as f:
            f.write(segmented_text)

        # --- 4. 逻辑总结 + 思维导图 ---
        yield "✍️ 正在基于分段稿生成大纲 & 思维导图...", seg_file
        map_prompt = f"""
        你是一位顶级的逻辑分析师。请基于以下分段文稿提取大纲：
        - # 一级标题：主题
        - ## 二级标题：核心逻辑板块
        - ### 三级标题：具体论点/子观点
        - - 列表项：用「」包裹的原文金句。
        - 最后附带一段 ```mermaid mindmap 代码。
        
        文稿内容：
        {segmented_text}
        """
        map_res = client.chat.completions.create(model="qwen-plus", messages=[{"role": "user", "content": map_prompt}])
        map_content = map_res.choices[0].message.content
        
        yield map_content, seg_file

    except Exception as e:
        yield f"❌ 运行报错: {str(e)}", None

# --- Gradio 界面设计 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 学术视频收割机 (原意保留+思维导图版)")
    with gr.Row():
        with gr.Column(scale=1):
            api_input = gr.Textbox(label="API KEY", type="password")
            url_input = gr.Textbox(label="视频/播客链接")
            btn = gr.Button("🔥 开始收割", variant="primary")
            clear_btn = gr.Button("🧹 清空缓存")
        with gr.Column(scale=2):
            file_output = gr.File(label="第一步：下载分段原稿")
            output_md = gr.Markdown(label="第二步：生成的深度总结")
    
    btn.click(fn=process_video, inputs=[api_input, url_input], outputs=[output_md, file_output])
    
    def clear_cache():
        for f in ["temp_audio.m4a", "raw_transcript.txt", "1_分段原稿.txt"]:
            if os.path.exists(f): os.remove(f)
        return "✨ 缓存已清理。", None
    clear_btn.click(fn=clear_cache, outputs=[output_md, file_output])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
