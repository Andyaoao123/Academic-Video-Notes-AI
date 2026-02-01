import os, whisper, yt_dlp
from openai import OpenAI
from google.colab import files

def run_podcast_tool():
    # --- 配置区 ---
    API_KEY = "你的sk-key" 
    VIDEO_URL = "你的链接"
    
    try:
        print("📥 正在抓取音频...")
        if os.path.exists("temp_audio.m4a"): os.remove("temp_audio.m4a")
        ydl_opts = {'format': 'm4a/bestaudio/best', 'outtmpl': 'temp_audio.%(ext)s'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([VIDEO_URL])
        
        print("🎙️ Whisper 听写中...")
        model = whisper.load_model("base") 
        result = model.transcribe("temp_audio.m4a")
        
        raw_name = "1_原始全文稿.txt"
        with open(raw_name, "w", encoding="utf-8") as f:
            f.write(result['text'])
        files.download(raw_name)

        print("✍️ 正在剥离逻辑骨架...")
        client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        prompt = f"你是一位严谨的思维导图专家。请根据文稿提炼逻辑骨架。要求：1.严禁脑补；2.使用Markdown标题层级（# ## ###）；3.用短句总结。文稿：\n{result['text']}"
        
        # 这一行是刚才报错的地方，现在写成一行确保安全
        response = client.chat.completions.create(model="qwen-plus", messages=[{"role": "user", "content": prompt}])
        
        map_name = "2_文稿逻辑框架.md"
        with open(map_name, "w", encoding="utf-8") as f:
            f.write(response.choices[0].message.content)
        
        files.download(map_name)
        print("🎉 任务完成！")

    except Exception as e:
        print(f"❌ 运行报错: {e}")

if __name__ == "__main__":
    run_podcast_tool()
