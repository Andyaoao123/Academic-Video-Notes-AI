import os, json, time, requests, whisper
from google.colab import files

# 用户配置
VIDEO_URL = "在这里填入视频链接" 
MY_API_KEY = "在这里填入你的KEY" 

def run_process(video_url=None, api_key=None):
    # 如果外部传了参数，就用外部的；否则用文件顶部的
    url = video_url if video_url else VIDEO_URL
    key = api_key if api_key else MY_API_KEY
    
    print(f"🌍 正在处理视频: {url}")
    os.system(f'yt-dlp -x --audio-format mp3 --force-overwrites -o "temp_audio.mp3" "{url}"')
    
    if not os.path.exists("temp_audio.mp3"):
        print("❌ 下载失败！请检查链接或权限。")
        return

    print("🎙️ 正在转录...")
    model = whisper.load_model("base")
    raw_text = model.transcribe("temp_audio.mp3", fp16=False)["text"]
    
    # 备份并下载
    with open("1_备份.txt", "w") as f: f.write(raw_text)
    files.download("1_备份.txt")

    # AI 精修
    print("🧠 AI 精修中...")
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
    prompt = f"请整理以下内容：\n{raw_text}"
    

    
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
