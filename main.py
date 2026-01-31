# ==========================================
# Academic Video Notes AI - 一键化学术笔记
# ==========================================
import os, json, time, requests, whisper
from google.colab import files

# 用户配置
VIDEO_URL = "在这里填入视频链接" 
MY_API_KEY = "在这里填入你的KEY_务必保留引号" 

def main():
    # 1. 安装环境 (Colab 环境专用)
    print("📦 正在准备环境...")
    os.system("pip install -q openai-whisper yt-dlp")
    
    # 2. 下载音频
    print("🌍 正在抓取音频...")
    os.system(f'yt-dlp -x --audio-format mp3 -o "temp_audio.%(ext)s" "{VIDEO_URL}"')
    
    # 3. Whisper 转录
    print("🎙️ 正在识别语音...")
    model = whisper.load_model("base")
    raw_text = model.transcribe("temp_audio.mp3")["text"]
    
    # 4. Gemini 精修
    print("🧠 AI 正在排版...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={MY_API_KEY}"
    prompt = f"请将以下讲座内容整理成带有小标题、核心观点和Mermaid导图的代码：\n{raw_text}"
    
    response = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
    
    # 5. 保存并下载
    with open("笔记.md", "w", encoding="utf-8") as f:
        f.write(response.json()['candidates'][0]['content']['parts'][0]['text'])
    files.download("笔记.md")
    print("✅ 完成！")

if __name__ == "__main__":
    main()
