import os
import whisper
import yt_dlp
from openai import OpenAI
from google.colab import files

def run_podcast_tool():
    # ==========================================
    # 1. 核心配置区 (仅改这里)
    # ==========================================
    API_KEY = "你的sk-开头Key" 
    VIDEO_URL = "你的网页链接"
    
    # ==========================================
    # 2. 自动化执行逻辑
    # ==========================================
    try:
        # A. 抓取音频
        print("📥 正在抓取音频...")
        if os.path.exists("temp_audio.m4a"): os.remove("temp_audio.m4a")
        ydl_opts = {'format': 'm4a/bestaudio/best', 'outtmpl': 'temp_audio.%(ext)s'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([VIDEO_URL])
        
        # B. 语音转文字
        print("🎙️ Whisper 正在精准听写 (大约需要 3-8 分钟)...")
        model = whisper.load_model("base") 
        result = model.transcribe("temp_audio.m4a")
        
        # --- 导出 1: 1:1 原始文稿 ---
        raw_name = "1_原始全文稿.txt"
        with open(raw_name, "w", encoding="utf-8") as f:
            f.write(result['text'])
        print(f"✅ 已导出原始文稿：{raw_name}")
        files.download(raw_name)

        # C. 逻辑框架提取
        print("✍️ 通义千问正在剥离逻辑骨架...")
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # 强制 AI 尊重原稿，生成思维导图结构
        prompt = f"""
        你是一位极其严谨的思维导图专家。请根据提供的转录文稿，提炼出其背后的【逻辑骨架】。
        
        要求：
        1. 严禁脑补：只总结文稿中出现的观点，不要添加任何外部知识或解释。
        2. 树状结构：使用 Markdown 标题层级（# ## ###）来体现逻辑关系，确保可以直接导入思维导图软件。
        3. 提炼核心：不要大段摘录，用短句总结每一节的核心含义。
        4. 关键词汇：保留文稿中特有的专有名词或高频核心词。
        
        文稿：
        {result['text']}
        """
        
        response = client.chat.completions.
