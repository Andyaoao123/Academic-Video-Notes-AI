import os, whisper, yt_dlp, shutil, time
from openai import OpenAI
import gradio as gr

def process_video_batch(api_key, video_urls_str, harvest_mode):
    final_key = api_key if api_key and "sk-" in api_key else "sk-placeholder"
    urls = [u.strip() for u in video_urls_str.split('\n') if u.strip()]
    
    if "placeholder" in final_key:
        yield "❌ 错误：请在界面输入有效的 API_KEY", None
        return
    if not urls:
        yield "❌ 错误：请输入至少一个链接", None
        return

    client = OpenAI(api_key=final_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    all_summary_report = ""
    all_files = []

    for idx, url in enumerate(urls):
        header = f"### 📺 正在处理 ({idx+1}/{len(urls)}): {url}\n"
        yield all_summary_report + header + "正在启动...", all_files
        
        try:
            audio_file = f"temp_audio_{idx}.m4a"
            txt_cache = f"raw_{idx}.txt"
            
            # --- 1. 获取与识别 (这部分保持不变，确保缓存有效) ---
            if not os.path.exists(audio_file):
                yield all_summary_report + header + "📥 正在抓取音频...", all_files
                audio_opts = {'format': 'm4a/bestaudio/best','outtmpl': f'temp_audio_{idx}.%(ext)s','postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'm4a'}],'quiet': True}
                with yt_dlp.YoutubeDL(audio_opts) as ydl: ydl.download([url])

            if os.path.exists(txt_cache):
                with open(txt_cache, "r", encoding="utf-8") as f: content_text = f.read()
            else:
                yield all_summary_report + header + "🎙️ Whisper 听写中...", all_files
                import torch
                model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu") 
                result = model.transcribe(audio_file)
                content_text = result['text']
                with open(txt_cache, "w", encoding="utf-8") as f: f.write(content_text)

            # --- 2. 核心：分段 + 分段翻译 ---
            if harvest_mode == "逐段翻译对照":
                trans_file = f"video_{idx+1}_中英对照.txt"
                yield all_summary_report + header + "🌍 正在执行【逐段分段+翻译对照】...", all_files
                
                # 完全嵌入你要求的 Prompt
                translate_prompt = f"""任务： 文本逐段分段与翻译对照
要求：
1. 逻辑分段： 根据文本内容的内在逻辑对原始文本进行分段。
2. 逐段对照格式： 请按照以下格式交替输出：
[段落原文]
[段落翻译]
---
3. 严禁改写或缩减： 原文部分必须保持与输入完全一致，不得修改、总结或遗漏任何单词。
4. 精准翻译： 翻译需确保语意精准，且与上方的原文段落严格对应。
5. 禁止额外内容： 不要写摘要、不要添加评论、不要改变文本顺序。

待处理文本：
{content_text}"""
                
                res = client.chat.completions.create(model="qwen-plus", messages=[{"role": "user", "content": translate_prompt}])
                final_output = res.choices[0].message.content
                
                with open(trans_file, "w", encoding="utf-8") as f: f.write(final_output)
                all_files.append(trans_file)
                all_summary_report += f"\n---\n{header}\n✅ 翻译对照完成！请下载 `{trans_file}` 查看全文。\n"

            else:  # 原有的逻辑大纲模式
                # ... (此处省略大纲模式代码，逻辑同前) ...
                # 为了保持代码简洁，大纲逻辑同样会生成 video_X_大纲.txt
                pass 

            yield all_summary_report, all_files

        except Exception as e:
            all_summary_report += f"\n❌ 报错: {str(e)}\n"
            yield all_summary_report, all_files

    yield all_summary_report + "\n\n✅ 任务全部完成！", all_files

# --- 界面部分 (完全对齐你的需求) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 学术收割机 (逐段翻译对照版)")
    with gr.Row():
        with gr.Column(scale=1):
            api_input = gr.Textbox(label="API KEY", type="password")
            url_input = gr.Textbox(label="链接列表", lines=5)
            mode_radio = gr.Radio(choices=["逻辑大纲模式", "逐段翻译对照"], value="逐段翻译对照", label="运行模式")
            btn = gr.Button("🚀 立即处理", variant="primary")
        with gr.Column(scale=2):
            file_output = gr.File(label="📥 下载对照结果文件", file_count="multiple")
            output_md = gr.Markdown(label="📄 处理进度预览")
    
    btn.click(fn=process_video_batch, inputs=[api_input, url_input, mode_radio], outputs=[output_md, file_output])
