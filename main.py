import os, whisper, yt_dlp, shutil
from openai import OpenAI
from google.colab import files
import markdown
import gradio as gr
from IPython.display import display, HTML

# --- 外部接口区 ---
API_KEY = "sk-placeholder"
VIDEO_URL = "url-placeholder"

def process_video(api_key, video_url):
    """Gradio 调用的核心逻辑函数"""
    # 优先使用界面输入的 KEY
    final_key = api_key if api_key and "sk-" in api_key else API_KEY
    
    if "placeholder" in final_key:
        yield "❌ 错误：请在界面输入有效的 API_KEY"
        return
    if not video_url:
        yield "❌ 错误：请输入视频或播客链接"
        return

    try:
        # --- 1. 音频获取 (断点续传逻辑) ---
        audio_file = "temp_audio.m4a"
        if os.path.exists(audio_file):
            yield "📁 检测到本地已存在音频，跳过下载步骤..."
        else:
            yield "📥 正在从源站抓取音频 (这可能需要 1-2 分钟)..."
            audio_opts = {
                'format': 'm4a/bestaudio/best',
                'outtmpl': 'temp_audio.%(ext)s',
                'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'm4a'}],
                'quiet': True
            }
            with yt_dlp.YoutubeDL(audio_opts) as ydl:
                ydl.download([video_url])

        # --- 2. 语音识别 (缓存逻辑) ---
        txt_cache = "raw_transcript.txt"
        if os.path.exists(txt_cache):
            yield "📄 检测到已存在识别文本，跳过 Whisper 听写，直接进入 AI 处理..."
            with open(txt_cache, "r", encoding="utf-8") as f:
                content_text = f.read()
        else:
            yield "🎙️ Whisper 正在拼命听写 (1小时视频约需 5-10 分钟)..."
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = whisper.load_model("base", device=device) 
            result = model.transcribe(audio_file)
            content_text = result['text']
            # 写入缓存，防止下次崩溃重来
            with open(txt_cache, "w", encoding="utf-8") as f:
                f.write(content_text)
        
        # --- 3. 语义整形 ---
        yield "🔍 正在进行文本整形手术 (语义纠错 & 标点还原)..."
        client = OpenAI(api_key=final_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        correction_prompt = f"""
        你是一位精通中文语境的资深编辑。请对以下语音稿进行“深度重塑”：
        1. 【加标点与断句】：根据语义添加标点。严禁出现超过15字没有标点的长句。
        2. 【全能纠错】：自动修正谐音错误（如：笑船->哮喘, 邻气->灵性）。
        3. 【去除口语赘词】：过滤“那个、就是、然后、呃”等。
        4. 【分段】：每段只聚焦一个核心语义点。
        
        内容如下：
        {content_text}
        """
        
        corr_res = client.chat.completions.create(model="qwen-plus", messages=[{"role": "user", "content": correction_prompt}])
        corrected_text = corr_res.choices[0].message.content

        # --- 4. 深度逻辑大纲提取 ---
        yield "✍️ 正在生成带金句的深度逻辑大纲..."
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

        # --- 5. 保存并展示 ---
        filename = "深度逻辑大纲.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(map_content)
        
        yield map_content

    except Exception as e:
        yield f"❌ 运行报错: {str(e)}"

# --- Gradio 界面设计 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 学术视频/播客深度收割机 (断点续传版)")
    with gr.Row():
        with gr.Column():
            api_input = gr.Textbox(label="DeepSeek API KEY", placeholder="sk-...", type="password")
            url_input = gr.Textbox(label="视频/播客链接", placeholder="支持 B站、YouTube、小宇宙...")
            with gr.Row():
                btn = gr.Button("🔥 开始收割", variant="primary")
                clear_btn = gr.Button("🧹 清空缓存 (换视频点这个)")
        with gr.Column():
            output = gr.Markdown(label="生成的深度大纲")
    
    # 绑定开始运行
    btn.click(fn=process_video, inputs=[api_input, url_input], outputs=output)
    
    # 绑定清空缓存
    def clear_cache():
        for f in ["temp_audio.m4a", "raw_transcript.txt"]:
            if os.path.exists(f): os.remove(f)
        return "✨ 缓存已清理，下次运行将重新下载音频和识别文字。"
    clear_btn.click(fn=clear_cache, outputs=output)

if __name__ == "__main__":
    # share=True 会生成公网链接
    demo.launch(share=True, debug=True)
