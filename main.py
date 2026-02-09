import os, whisper, yt_dlp, shutil, time
from openai import OpenAI
import gradio as gr
from pydub import AudioSegment

def call_ai_pipeline(client, harvest_mode, text_content):
    """提取出的 AI 处理核心逻辑"""
    # Step B: 标点还原
    clean_p = f"你是一位文字整理师。请对以下原始语音文本进行【标点还原】和【逻辑分段】，严禁删减或润色原文词语：\n\n{text_content}"
    clean_res = client.chat.completions.create(model="qwen-turbo", messages=[{"role": "user", "content": clean_p}])
    segmented_text = clean_res.choices[0].message.content

    # Step C: 根据模式输出
    if harvest_mode == "逐段翻译对照":
        trans_p = f"任务：文本逐段翻译对照。要求：格式为 [原文段落]\n[翻译段落]\n---。请翻译以下已分段文本，确保翻译精准且不改写原文：\n\n{segmented_text}"
        res = client.chat.completions.create(model="qwen-plus", messages=[{"role": "user", "content": trans_p}])
        return res.choices[0].message.content
    elif harvest_mode == "逻辑大纲模式":
        map_p = f"你是一位逻辑分析师。请基于以下分段文稿提取大纲和mermaid mindmap代码，核心论点用「」包裹：\n\n{segmented_text}"
        res = client.chat.completions.create(model="qwen-plus", messages=[{"role": "user", "content": map_p}])
        return res.choices[0].message.content
    else:
        return segmented_text

def process_all_in_one(api_key, input_content, harvest_mode):
    final_key = api_key if api_key and "sk-" in api_key else "sk-placeholder"
    if "placeholder" in final_key:
        yield "❌ 错误：请在界面输入有效的 API_KEY", None
        return
    if not input_content.strip():
        yield "❌ 错误：输入内容不能为空", None
        return

    client = OpenAI(api_key=final_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    all_summary_report = ""
    all_files = []
    lines = [l.strip() for l in input_content.split('\n') if l.strip()]

    if harvest_mode == "雅思作文教练":
        yield "✍️ 考官正在研读你的大作...", None
        ielts_p = f"任务：雅思前考官深度批改...\n{input_content}" 
        try:
            res = client.chat.completions.create(model="qwen-plus", messages=[{"role": "user", "content": ielts_p}])
            feedback = res.choices[0].message.content
            f_name = "雅思作文批改报告.txt"
            with open(f_name, "w", encoding="utf-8") as f: f.write(feedback)
            yield feedback, [f_name]
        except Exception as e: yield f"❌ 批改失败: {str(e)}", None
        return

    for idx, content in enumerate(lines):
        header = f"### 📺 正在处理 ({idx+1}/{len(lines)}): {content}\n"
        yield all_summary_report + header + "正在检查/准备资源...", all_files
        
        try:
            audio_file = f"temp_audio_{idx}.mp3"
            
            # 【核心改进 1】：识别本地/Drive 路径并自动转码
            if content.startswith("/content/"):
                if not os.path.exists(audio_file):
                    yield all_summary_report + header + "📂 检测到路径，正在转换音轨...", all_files
                    raw_data = AudioSegment.from_file(content)
                    raw_data.export(audio_file, format="mp3")
            
            # 【核心改进 2】：增加 B 站防爬补丁下载
            elif not os.path.exists(audio_file):
                yield all_summary_report + header + "🌐 正在启动网络下载 (带防爬补丁)...", all_files
                opts = {
                    'format': 'mp3/bestaudio/best',
                    'outtmpl': f'temp_audio_{idx}.%(ext)s',
                    'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3'}],
                    'quiet': True,
                    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
                    'referer': 'https://www.bilibili.com/',
                    'nocheckcertificate': True
                }
                with yt_dlp.YoutubeDL(opts) as ydl: ydl.download([content])

            # 2. 解析音频
            audio_data = AudioSegment.from_file(audio_file)
            duration_mins = len(audio_data) / (60 * 1000)
            chunk_length = 20 * 60 * 1000
            chunks = [audio_data[i:i + chunk_length] for i in range(0, len(audio_data), chunk_length)]
            
            video_combined_text = ""
            yield all_summary_report + header + f"📦 全长 {duration_mins:.1f} 分钟，共 {len(chunks)} 段。检查断点...", all_files

            # 【核心改进 3】：智能检测 GPU，挂了就自动换 CPU
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = whisper.load_model("base", device=device)

            for c_idx, chunk in enumerate(chunks):
                tmp_out = f"video_{idx+1}_Part_{c_idx+1}.txt"
                segment_divider = f"\n\n--- 📜 第 {c_idx+1} 部分 (约 {c_idx*20}-{(c_idx+1)*20}min) ---\n\n"

                if os.path.exists(tmp_out):
                    yield all_summary_report + header + f"⏩ 加载已存断点: 第 {c_idx+1}/{len(chunks)} 段...", all_files
                    with open(tmp_out, "r", encoding="utf-8") as f:
                        processed_chunk_text = f.read()
                    video_combined_text += segment_divider + processed_chunk_text
                    if tmp_out not in all_files: all_files.append(tmp_out)
                    continue 

                chunk_filename = f"temp_{idx}_{c_idx+1}.mp3"
                chunk.export(chunk_filename, format="mp3")
                
                yield all_summary_report + header + f"🎙️ [第{c_idx+1}段] Whisper 听写中 ({device})...", all_files
                raw_chunk_text = model.transcribe(chunk_filename)['text']
                
                yield all_summary_report + header + f"✍️ [第{c_idx+1}段] AI 处理中...", all_files
                processed_chunk_text = call_ai_pipeline(client, harvest_mode, raw_chunk_text)
                
                with open(tmp_out, "w", encoding="utf-8") as f: f.write(processed_chunk_text)
                video_combined_text += segment_divider + processed_chunk_text
                all_files.append(tmp_out)
                
                yield all_summary_report + header + video_combined_text, all_files

            final_out_f = f"video_{idx+1}_完整收割稿.txt"
            with open(final_out_f, "w", encoding="utf-8") as f: f.write(video_combined_text)
            all_files.append(final_out_f)
            all_summary_report += f"\n---\n{header}\n✅ 处理完成！\n"
            yield all_summary_report, all_files

        except Exception as e:
            all_summary_report += f"\n❌ 报错: {str(e)}\n"
            yield all_summary_report, all_files

    yield all_summary_report + "\n\n✅ 任务全部完成！", all_files

# --- 界面部分 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 学术多功能收割机 (全能路径识别版)")
    with gr.Row():
        with gr.Column(scale=1):
            api_input = gr.Textbox(label="API KEY", type="password")
            input_box = gr.Textbox(label="输入区域", lines=8, placeholder="填链接 或 填 /content/ 开头的本地路径")
            mode_radio = gr.Radio(
                choices=["逻辑大纲模式", "逐段整理(不翻译)", "逐段翻译对照", "雅思作文教练"], 
                value="逐段整理(不翻译)", 
                label="选择作战模式"
            )
            btn = gr.Button("🚀 立即处理", variant="primary")
            clear_btn = gr.Button("🧹 清理缓存文件")
        with gr.Column(scale=2):
            file_output = gr.File(label="📥 结果下载", file_count="multiple")
            output_md = gr.Markdown(label="📄 实时预览")
    
    btn.click(fn=process_all_in_one, inputs=[api_input, input_box, mode_radio], outputs=[output_md, file_output])
    
    def clear():
        for f in os.listdir():
            if f.startswith(("temp_", "video_", "cache_")) or f.endswith((".mp3", ".txt", ".m4a")):
                try: os.remove(f)
                except: pass
        return "✨ 缓存已清空，可以开始处理新文件了。", None
    clear_btn.click(fn=clear, outputs=[output_md, file_output])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
