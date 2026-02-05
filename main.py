import os, whisper, yt_dlp, shutil, time
from openai import OpenAI
import gradio as gr

def process_video_batch(api_key, video_urls_str, harvest_mode):
    """批量处理逻辑：循环每一个链接"""
    final_key = api_key if api_key and "sk-" in api_key else "sk-placeholder"
    
    # 1. 拆分链接
    urls = [u.strip() for u in video_urls_str.split('\n') if u.strip()]
    
    if "placeholder" in final_key:
        yield "❌ 错误：请在界面输入有效的 API_KEY", None
        return
    if not urls:
        yield "❌ 错误：请输入至少一个链接（每行一个）", None
        return

    client = OpenAI(api_key=final_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    all_summary_report = ""
    all_segmented_files = []

    for idx, url in enumerate(urls):
        header = f"### 📺 正在处理 ({idx+1}/{len(urls)}): {url}\n"
        yield all_summary_report + header + "正在启动...", all_segmented_files
        
        try:
            # 2. 文件名初始化
            audio_file = f"temp_audio_{idx}.m4a"
            txt_cache = f"raw_{idx}.txt"
            
            # --- A. 获取音频 ---
            if not os.path.exists(audio_file):
                yield all_summary_report + header + "📥 正在抓取音频...", all_segmented_files
                audio_opts = {
                    'format': 'm4a/bestaudio/best',
                    'outtmpl': f'temp_audio_{idx}.%(ext)s',
                    'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'm4a'}],
                    'quiet': True
                }
                with yt_dlp.YoutubeDL(audio_opts) as ydl:
                    ydl.download([url])

            # --- B. 语音识别 ---
            if os.path.exists(txt_cache):
                with open(txt_cache, "r", encoding="utf-8") as f:
                    content_text = f.read()
            else:
                yield all_summary_report + header + "🎙️ Whisper 听写中...", all_segmented_files
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = whisper.load_model("base", device=device) 
                result = model.transcribe(audio_file)
                content_text = result['text']
                with open(txt_cache, "w", encoding="utf-8") as f:
                    f.write(content_text)

            # --- C. 分不同模式进行 AI 处理 ---
            if harvest_mode == "逻辑大纲模式":
                seg_file = f"video_{idx+1}_分段原稿.txt"
                yield all_summary_report + header + "📑 正在原意分段...", all_segmented_files
                
                # 步骤 1: 分段
                split_prompt = f"你是一位专业的速记员。请对以下文稿进行【仅分段与加标点】处理，严禁删减、改动或润色原文任何词语：\n\n{content_text}"
                split_res = client.chat.completions.create(model="qwen-turbo", messages=[{"role": "user", "content": split_prompt}])
                segmented_text = split_res.choices[0].message.content
                with open(seg_file, "w", encoding="utf-8") as f:
                    f.write(segmented_text)
                all_segmented_files.append(seg_file)

                # 步骤 2: 大纲
                yield all_summary_report + header + "✍️ 正在提炼大纲...", all_segmented_files
                map_prompt = f"你是一位顶级的逻辑分析师。请基于以下分段文稿提取大纲和 mermaid mindmap 代码。要求：严格保留核心论点，用「」包裹金句。\n\n文稿内容：\n{segmented_text}"
                map_res = client.chat.completions.create(model="qwen-plus", messages=[{"role": "user", "content": map_prompt}])
                map_content = map_res.choices[0].message.content
                all_summary_report += f"\n---\n{header}\n{map_content}\n"

            else:  # 翻译模式
                trans_file = f"video_{idx+1}_中英对照翻译.txt"
                yield all_summary_report + header + "🌍 正在进行逐段中英对照...", all_segmented_files
                
                # 使用你提供的优质翻译 Prompt
                translate_prompt = f"""任务： 文本逐段分段与翻译对照要求：
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
                
                trans_res = client.chat.completions.create(model="qwen-plus", messages=[{"role": "user", "content": translate_prompt}])
                translated_content = trans_res.choices[0].message.content
                
                with open(trans_file, "w", encoding="utf-8") as f:
                    f.write(translated_content)
                all_segmented_files.append(trans_file)
                all_summary_report += f"\n---\n{header}\n✅ 翻译完成，请在下方下载文件查看详情。\n"

            yield all_summary_report, all_segmented_files

        except Exception as e:
            error_msg = f"\n❌ 视频 {idx+1} 运行报错: {str(e)}\n"
            all_summary_report += error_msg
            yield all_summary_report, all_segmented_files

    yield all_summary_report + "\n\n✅ 所有任务处理完成！", all_segmented_files

# --- Gradio 界面设计 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 学术视频多功能收割机 (ADHD友好/不阉割版)")
    with gr.Row():
        with gr.Column(scale=1):
            api_input = gr.Textbox(label="API KEY", type="password")
            url_input = gr.Textbox(label="视频链接 (每行一个)", lines=5, placeholder="粘贴多个链接，一行一个...")
            # --- 模式切换按钮 ---
            mode_radio = gr.Radio(
                choices=["逻辑大纲模式", "逐段翻译对照"], 
                value="逻辑大纲模式", 
                label="选择收割模式"
            )
            btn = gr.Button("🚀 开始收割", variant="primary")
            clear_btn = gr.Button("🧹 清空所有缓存")
        with gr.Column(scale=2):
            file_output = gr.File(label="📥 下载生成的文件", file_count="multiple")
            output_md = gr.Markdown(label="📄 汇总状态报告")
    
    # 绑定点击事件，注意增加了 mode_radio 作为输入
    btn.click(fn=process_video_batch, inputs=[api_input, url_input, mode_radio], outputs=[output_md, file_output])
    
    def clear_all():
        files_to_delete = [f for f in os.listdir() if f.endswith((".m4a", ".txt", ".md"))]
        for f in files_to_delete: os.remove(f)
        return "✨ 所有缓存文件已清理。", None
    clear_btn.click(fn=clear_all, outputs=[output_md, file_output])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
