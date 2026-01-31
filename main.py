# ==========================================
# Academic Video Notes AI - 增强备份版
# ==========================================
import os, json, time, requests, whisper
from google.colab import files

# 用户配置
VIDEO_URL = "在这里填入视频链接" 
MY_API_KEY = "在这里填入你的KEY" 

def main():
    # 1. 环境准备
    print("📦 正在准备环境...")
    os.system("pip install -q openai-whisper yt-dlp")
    
    # 2. 下载音频
    print("🌍 正在抓取音频...")
    os.system(f'yt-dlp -x --audio-format mp3 -o "temp_audio.%(ext)s" "{VIDEO_URL}"')
    
    # 3. Whisper 转录
    print("🎙️ 正在识别语音 (这一步最耗时，请稍候)...")
    model = whisper.load_model("base")
    raw_text = model.transcribe("temp_audio.mp3")["text"]
    
    # --- ✨ 新增备份逻辑 ✨ ---
    backup_file = "1_原始转录文本_备份.txt"
    with open(backup_file, "w", encoding="utf-8") as f:
        f.write(raw_text)
    print(f"💾 备份成功！原始文本已存入: {backup_file}")
    files.download(backup_file) # 先把原始文件弹出来，确保安全
    # --------------------------

    # 4. Gemini 精修
    print("🧠 AI 正在尝试排版精修...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={MY_API_KEY}"
    prompt = f"请将以下讲座内容整理成带有小标题、核心观点和Mermaid导图的代码：\n{raw_text}"
    
    try:
        response = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=60)
        if response.status_code == 200:
            polished_text = response.json()['candidates'][0]['content']['parts'][0]['text']
            
            # 保存精修版
            output_file = "2_AI精修学术笔记.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(polished_text)
            
            print(f"✅ AI 精修完成！")
            files.download(output_file)
        else:
            print(f"⚠️ AI 接口报错 (错误码 {response.status_code})，但你的原始备份已下载。")
    except Exception as e:
        print(f"⚠️ 联网精修失败: {e}。请检查 API Key 或网络。")

if __name__ == "__main__":
    main()
