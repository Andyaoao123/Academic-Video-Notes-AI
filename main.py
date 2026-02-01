import os, whisper, yt_dlp
from openai import OpenAI
from google.colab import files

# --- 外部接口区：把变量放外面，README 脚本才改得到 ---
API_KEY = "sk-placeholder"
VIDEO_URL = "url-placeholder"

def run_podcast_tool():
    """
    核心执行函数。它会自动读取上面定义的 API_KEY 和 VIDEO_URL。
    """
    if "placeholder" in API_KEY or "placeholder" in VIDEO_URL:
        print("❌ 错误：请先在 Colab 中设置 main.API_KEY 和 main.VIDEO_URL")
        return

    try:
        # A. 抓取音频
        print(f"📥 正在抓取音频: {VIDEO_URL}")
        if os.path.exists("temp_audio.m4a"): os.remove("temp_audio.m4a")
        ydl_opts = {'format': 'm4a/bestaudio/best', 'outtmpl': 'temp_audio.%(ext)s'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([VIDEO_URL])
        
        # B. 语音转文字
        print("🎙️ Whisper 正在精准听写...")
        model = whisper.load_model("base") 
        result = model.transcribe("temp_audio.m4a")
        
        # 导出 1: 原始文稿
        raw_name = "1_原始全文稿.txt"
        with open(raw_name, "w", encoding="utf-8") as f:
            f.write(result['text'])
        files.download(raw_name)

        # C. 逻辑框架提取
        print("✍️ 通义千问正在剥离逻辑骨架...")
        client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        prompt = f"你是一位严谨的思维导图专家。请根据文稿提炼逻辑骨架。要求：1.严禁脑补；2.使用Markdown标题层级（# ## ###）；3.用短句总结。文稿：\n{result['text']}"
        
        response = client.chat.completions.create(model="qwen-plus", messages=[{"role": "user", "content": prompt}])
        
        # 导出 2: 逻辑框架
        map_name = "2_文稿逻辑框架.md"
        with open(map_name, "w", encoding="utf-8") as f:
            f.write(response.choices[0].message.content)
        
        files.download(map_name)
        print("🎉 任务完成！请检查浏览器下载提示。")

    except Exception as e:
        print(f"❌ 运行报错: {e}")

if __name__ == "__main__":
    run_podcast_tool()
