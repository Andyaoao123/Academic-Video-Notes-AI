import os, whisper, yt_dlp, shutil, time
from openai import OpenAI
import gradio as gr

def process_all_in_one(api_key, input_content, harvest_mode):
    """
    三合一流水线架构：
    1. 雅思模式：直接 Agent 批改
    2. 视频模式：Whisper识别 -> B.标点整理与逻辑分段 -> C.分发(翻译/大纲/纯整理)
    """
    final_key = api_key if api_key and "sk-" in api_key else "sk-placeholder"
    
    # 基础校验
    if "placeholder" in final_key:
        yield "❌ 错误：请在界面输入有效的 API_KEY", None
        return
    if not input_content.strip():
        yield "❌ 错误：输入内容不能为空", None
        return

    client = OpenAI(api_key=final_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    all_summary_report = ""
    all_files = []
    
    # 拆分多行链接
    lines = [l.strip() for l in input_content.split('\n') if l.strip()]

    # --- 场景 A：雅思作文教练 (Agent 模式) ---
    if harvest_mode == "雅思作文教练":
        yield "✍️ 考官正在研读你的大作，准备给出即时反馈...", None
        ielts_p = f"""任务：请作为一名拥有15年经验的雅思资深前考官，对以下作文进行深度批改。
要求：
1. 【预估分值】：给出总分及四个单项（TR/CC/LR/GRA）的预估分。
2. 【逻辑漏洞】：指出文章论证不严密或衔接突兀的地方。
3. 【词汇升级】：找出文中3-5个高级学术词汇替换方案。
4. 【语法纠错】：修正错误并解释原因。
5. 【高分范文】：提供一份 8 分参考范文。
语气：幽默且犀利。

待批改作文：
{input_content}"""
        
        try:
            res = client.chat.completions.create(model="qwen-plus", messages=[{"role": "user", "content": ielts_p}])
            feedback = res.choices[0].message.content
            f_name = "雅思作文批改报告.txt"
            with open(f_name, "w", encoding="utf-8") as f: f.write(feedback)
            yield feedback, [f_name]
        except Exception as e:
            yield f"❌ 批改失败: {str(e)}", None
        return

    # --- 场景 B：视频处理流水线 ---
    for idx, url in enumerate(lines):
        header = f"### 📺 正在处理 ({idx+1}/{len(lines)}): {url}\n"
        yield all_summary_report + header + "正在启动...", all_files
        
        try:
            audio_file = f"temp_audio_{idx}.m4a"
            txt_cache = f"raw_{idx}.txt"
            
            # 1. 语音识别 (Whisper)
            if not os.path.exists(audio_file):
                yield all_summary_report + header + "📥 正在抓取音频...", all_files
                opts = {
                    'format': 'm4a/bestaudio/best',
                    'outtmpl': f'temp_audio_{idx}.%(ext)s',
                    'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'm4a'}],
                    'quiet': True
                }
                with yt_dlp.YoutubeDL(opts) as ydl: ydl.download([url])

            if os.path.exists(txt_cache):
                with open(txt_cache, "r", encoding="utf-8") as f: content_text = f.read()
            else:
                yield all_summary_report + header + "🎙️ Whisper 听写中...", all_files
                import torch
                model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu") 
                content_text = model.transcribe(audio_file)['text']
                with open(txt_cache, "w", encoding="utf-8") as f: f.write(content_text)

            # 2. 流水线核心：Step B - 标点还原与逻辑分段 (解决“没标点”的痛点)
            yield all_summary_report + header + "✍️ 正在进行标点还原与逻辑分段...", all_files
            clean_p = f"你是一位文字整理师。请对以下原始语音文本进行【标点还原】和【逻辑分段】，严禁删减或润色原文词语：\n\n{content_text}"
            clean_res = client.chat.completions.create(model="qwen-turbo", messages=[{"role": "user", "content": clean_p}])
            segmented_text = clean_res.choices[0].message.content

            # 3. 流水线核心：Step C - 根据模式输出最终成品
            if harvest_mode == "逐段翻译对照":
                yield all_summary_report + header + "🌍 正在基于整理稿进行对照翻译...", all_files
                trans_p = f"任务：文本逐段翻译对照。要求：格式为 [原文段落]\n[翻译段落]\n---。请翻译以下已分段文本，确保翻译精准且不改写原文：\n\n{segmented_text}"
                res = client.chat.completions.create(model="qwen-plus", messages=[{"role": "user", "content": trans_p}])
                final_result = res.choices[0].message.content
                out_f = f"video_{idx+1}_中英对照.txt"
            
            elif harvest_mode == "逻辑大纲模式":
                yield all_summary_report + header + "📊 正在提炼大纲与思维导图...", all_files
                map_p = f"你是一位逻辑分析师。请基于以下分段文稿提取大纲和mermaid mindmap代码，核心论点用「」包裹：\n\n{segmented_text}"
                res = client.chat.completions.create(model="qwen-plus", messages=[{"role": "user", "content": map_p}])
                final_result = res.choices[0].message.content
                out_f = f"video_{idx+1}_逻辑大纲.txt"
            
            else: # 逐段整理 (不翻译)
                final_result = segmented_text
                out_f = f"video_{idx+1}_纯净整理稿.txt"

            # 存盘并更新 UI
            with open(out_f, "w", encoding="utf-8") as f: f.write(final_result)
            all_files.append(out_f)
            all_summary_report += f"\n---\n{header}\n{final_result}\n"
            yield all_summary_report, all_files

        except Exception as e:
            all_summary_report += f"\n❌ 报错: {str(e)}\n"
            yield all_summary_report, all_files

    yield all_summary_report + "\n\n✅ 所有任务处理完成！", all_files

# --- Gradio 界面设计 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 学术/写作多功能全能收割机")
    with gr.Row():
        with gr.Column(scale=1):
            api_input = gr.Textbox(label="API KEY", type="password")
            input_box = gr.Textbox(label="输入区域", lines=8, placeholder="视频填链接(一行一个)；作文填原文内容")
            mode_radio = gr.Radio(
                choices=["逻辑大纲模式", "逐段整理(不翻译)", "逐段翻译对照", "雅思作文教练"], 
                value="逐段整理(不翻译)", 
                label="选择作战模式"
            )
            btn = gr.Button("🚀 立即处理", variant="primary")
            clear_btn = gr.Button("🧹 清理缓存文件")
        with gr.Column(scale=2):
            file_output = gr.File(label="📥 下载生成的文件", file_count="multiple")
            output_md = gr.Markdown(label="📄 实时反馈面板")
    
    btn.click(fn=process_all_in_one, inputs=[api_input, input_box, mode_radio], outputs=[output_md, file_output])
    
    def clear():
        for f in os.listdir():
            if f.endswith((".m4a", ".txt", ".mp4")): os.remove(f)
        return "✨ 目录已净化，缓存已清空。", None
    clear_btn.click(fn=clear, outputs=[output_md, file_output])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
