import os, json, time, requests, whisper, subprocess
from google.colab import files

# 用户配置
VIDEO_URL = "在这里填入视频链接" 
MY_API_KEY = "在这里填入你的KEY" 

def main():
    print("📦 正在准备环境...")
    os.system("pip install -q openai-whisper yt-dlp")
    
    # 1. 强力下载逻辑
    print(f"🌍 正在尝试抓取: {VIDEO_URL}")
    # 使用 subprocess 捕获下载错误，并添加 --force-overwrites 确保覆盖
    download_cmd = f'yt-dlp -x --audio-format mp3 --force-overwrites -o "temp_audio.%(ext)s" "{VIDEO_URL}"'
    result = os.system(download_cmd)
    
    if result != 0 or not os.path.exists("temp_audio.mp3"):
        print("❌ 下载失败！请检查视频链接是否正确，或该视频是否需要登录才能观看。")
        return

    # 2. Whisper 转录
    print("🎙️ 正在识别语音 (请耐心等待)...")
    try:
        model = whisper.load_model("base")
        # 显式指定识别 temp_audio.mp3
        transcribe_result = model.transcribe("temp_audio.mp3", fp16=False)
        raw_text = transcribe_result["text"]
    except Exception as e:
        print(f"🎙️ 转录过程出错: {e}")
        return
    
    # 3. 备份
    backup_file = "1_原始转录文本_备份.txt"
    with open(backup_file, "w", encoding="utf-8") as f:
        f.write(raw_text)
    print(f"💾 原始文本已备份至: {backup_file}")
    files.download(backup_file)

    # 4. AI 精修
    print("🧠 AI 正在排版精修...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={MY_API_KEY}"
    prompt = f"请将以下讲座内容整理成带有小标题、核心观点和Mermaid导图的代码：\n{raw_text}"
    
    try:
        response = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=60)
        if response.status_code == 200:
            output_file = "2_AI精修学术笔记.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(response.json()['candidates'][0]['content']['parts'][0]['text'])
            print(f"✅ AI 精修完成！")
            files.download(output_file)
        else:
            print(f"⚠️ AI 接口返回错误: {response.status_code}")
    except Exception as e:
        print(f"⚠️ 联网精修失败: {e}")

if __name__ == "__main__":
    main()
