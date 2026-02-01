import os
import whisper
import yt_dlp
from openai import OpenAI
from google.colab import files

def run_podcast_tool():
    # ==========================================
    # 1. 核心配置区 (在这里填入你的信息)
    # ==========================================
    API_KEY = "你的sk-开头Key" 
    VIDEO_URL = "你的网页链接"
    
    # ==========================================
    # 2. 核心逻辑区
    # ==========================================
    try:
        # A. 抓取音频
        print("📥 正在下载音频...")
        if os.path.exists("temp_audio.m4a"): os.remove("temp_audio.m4a")
        ydl_opts = {'format': 'm4a/bestaudio/best', 'outtmpl': 'temp_audio.%(ext)s'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([VIDEO_URL])
        
        # B. 语音转文字
        print("🎙️ Whisper 正在拼命听写 (大约需要 3-8 分钟)...")
        model = whisper.load_model("base") 
        result = model.transcribe("temp_audio.m4a")
        
        # --- 导出文件1: 原始文稿 ---
        raw_name = "1_原始全文稿.txt"
        with open(raw_name, "w", encoding="utf-8") as f:
            f.write(result['text'])
        print(f"✅ 已生成原始文稿：{raw_name}")
        files.download(raw_name)

        # C. 大模型精修
        print("✍️ 通义千问正在整理深度笔记...")
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        prompt = f"""
        你是一位顶级的学术助教。请将以下播客转录文稿整理为深度研读笔记。
        要求：
        1. 核心主题：一句话总结。
        2. 逻辑拆解：提取 3-5 个核心观点并展开。
        3. 费曼学习：用“5岁小孩能听懂”的话解释其中最难的概念。
        4. 金句摘录：摘选 3 句最有启发的话。
        
        文稿：
        {result['text']}
        """
        
        response = client.chat.completions.create(
            model="qwen-plus", 
            messages=[{"role": "user", "content": prompt}]
        )
        
        # --- 导出文件2: 精修笔记 ---
        note_name = "2_千问精修笔记.md"
        with open(note_name, "w", encoding="utf-8") as f:
            f.write(response.choices[0].message.content)
        
        print(f"✅ 已生成精修笔记：{note_name}")
        files.download(note_name)
        print("🎉 全部任务已完成！请查看浏览器下载记录。")

    except Exception as e:
        print(f"❌ 运行报错: {e}")

# 执行程序
if __name__ == "__main__":
    run_podcast_tool()
