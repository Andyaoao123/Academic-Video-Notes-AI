import os, whisper, yt_dlp, shutil
from openai import OpenAI
from google.colab import files
import markdown
from IPython.display import display, HTML

# --- 外部接口区 ---
API_KEY = "sk-placeholder"
VIDEO_URL = "url-placeholder"

def run_podcast_tool():
    if "placeholder" in API_KEY or "placeholder" in VIDEO_URL:
        print("❌ 错误：请先设置 API_KEY 和 VIDEO_URL")
        return

    try:
        content_text = ""
        
        # 1. 尝试极速抓取字幕
        print(f"🔍 正在尝试从源站提取现成字幕...")
        ydl_opts_subs = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['zh-Hans', 'zh-CN', 'zh', 'en'],
            'outtmpl': 'subtitle_file',
            'quiet': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts_subs) as ydl:
                info = ydl.extract_info(VIDEO_URL, download=True)
                if 'requested_subtitles' in info and info['requested_subtitles']:
                    print("✅ 成功获取在线字幕！正在闪电提取...")
                else:
                    print("ℹ️ 未检测到外挂字幕。")
        except:
            pass

        # 2. 语音识别保底 (配合 GPU 提速)
        if not content_text:
            print(f"📥 正在获取音频并启动识别流程 (1小时视频预计 5-10 分钟)...")
            audio_opts = {
                'format': 'm4a/bestaudio/best',
                'outtmpl': 'temp_audio.%(ext)s',
                'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'm4a'}],
                'quiet': True
            }
            with yt_dlp.YoutubeDL(audio_opts) as ydl:
                ydl.download([VIDEO_URL])
            
            # 使用 base 模型，自动检测 GPU
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = whisper.load_model("base", device=device) 
            result = model.transcribe("temp_audio.m4a")
            content_text = result['text']
        
        # 3. 语义整形
        print("🔍 正在进行文本整形手术 (语义纠错 & 标点还原)...")
        client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        correction_prompt = f"""
        你是一位精通中文语境的资深编辑。请对以下语音稿进行“深度重塑”：
        1. 【加标点与断句】：根据语义添加标点。严禁出现超过15字没有标点的长句，确保节奏感。
        2. 【全能纠错】：结合上下文，自动修正谐音错误（如：笑船->哮喘, 完晒->完赛, 邻气->灵性, 墨生->默生）。
        3. 【去除口语赘词】：过滤“那个、就是、然后、呃、我的话、对吧”等填充词。
        4. 【分段】：每段只聚焦一个核心语义点。
        
        内容如下：
        {content_text}
        """
        
        corr_res = client.chat.completions.create(model="qwen-plus", messages=[{"role": "user", "content": correction_prompt}])
        corrected_text = corr_res.choices[0].message.content

        # 4. 深度逻辑大纲提取
        print("✍️ 正在生成带金句的深度逻辑大纲...")
        map_prompt = f"""
        你是一位顶级的逻辑分析师。请基于以下精排文稿提取大纲：
        - # 一级标题：主题
        - ## 二级标题：核心逻辑板块
        - ### 三级标题：具体论点/子观点
        - - 列表项：用「」包裹的原文核心金句。
        
        文稿内容：
        {corrected_text}
        """
        map_res = client.chat.completions.create(model="qwen-plus", messages=[{"role": "user", "content": map_prompt}])
        map_content = map_res.choices[0].message.content

        # 5. 下载、预览与网盘备份
        filename1 = "1_精排文稿.txt"
        filename2 = "2_深度逻辑大纲.md"
        
        with open(filename1, "w", encoding="utf-8") as f: f.write(corrected_text)
        with open(filename2, "w", encoding="utf-8") as f: f.write(map_content)
        
        # 自动触发浏览器下载
        files.download(filename1)
        files.download(filename2)

        # 核心：Google Drive 自动备份逻辑
        drive_path = "/content/drive/MyDrive/AI_Notes/"
        if os.path.exists("/content/drive"):
            if not os.path.exists(drive_path):
                os.makedirs(drive_path)
            shutil.copy(filename1, os.path.join(drive_path, filename1))
            shutil.copy(filename2, os.path.join(drive_path, filename2))
            print(f"💾 备份成功！文件已存入 Google Drive: {drive_path}")
        else:
            print("💡 提示：未挂载 Google Drive，文件仅保存在临时会话中。")

        print("\n--- 📝 实时预览 ---")
        display(HTML(f"<div style='background:#f9f9f9; padding:20px; border-radius:12px; border:1px solid #ddd; line-height:1.8;'>{markdown.markdown(map_content)}</div>"))

    except Exception as e:
        print(f"❌ 运行报错: {e}")

if __name__ == "__main__":
    run_podcast_tool()


# --- Gradio 界面设计 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 学术视频/播客深度收割机")
    with gr.Row():
        with gr.Column():
            api_input = gr.Textbox(label="DeepSeek API KEY", placeholder="sk-...", type="password")
            url_input = gr.Textbox(label="视频/播客链接", placeholder="支持 B站、YouTube、小宇宙...")
            btn = gr.Button("🔥 开始收割 (建议开启 T4 GPU)", variant="primary")
        with gr.Column():
            output = gr.Markdown(label="生成的深度大纲")
    
    btn.click(fn=process_video, inputs=[api_input, url_input], outputs=output)

if __name__ == "__main__":
    # share=True 会生成一个公网链接，你可以发给手机或者朋友用
    demo.launch(share=True, debug=True)
