import os, whisper, yt_dlp
from openai import OpenAI
from google.colab import files

# --- 外部接口区 ---
API_KEY = "sk-placeholder"
VIDEO_URL = "url-placeholder"

def run_podcast_tool():
    if "placeholder" in API_KEY or "placeholder" in VIDEO_URL:
        print("❌ 错误：请先在 Colab 中设置 main.API_KEY 和 main.VIDEO_URL")
        return

    try:
        print(f"📥 正在抓取音频: {VIDEO_URL}")
        if os.path.exists("temp_audio.m4a"): os.remove("temp_audio.m4a")
        ydl_opts = {'format': 'm4a/bestaudio/best', 'outtmpl': 'temp_audio.%(ext)s'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([VIDEO_URL])
        
        print("🎙️ Whisper 正在精准听写...")
        model = whisper.load_model("base") 
        result = model.transcribe("temp_audio.m4a")
        
        raw_name = "1_原始全文稿.txt"
        with open(raw_name, "w", encoding="utf-8") as f:
            f.write(result['text'])
        files.download(raw_name)

        print("✍️ 正在进行深度逻辑剥离（包含核心金句）...")
        client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        # --- 强化的 Prompt 逻辑 ---
        prompt = f"""
        你是一位顶级的逻辑分析师。请将提供的转录文稿整理成一份【深度思维导图大纲】。
        
        要求如下：
        1. 结构化层级：
           - # 一级标题：播客核心主题
           - ## 二级标题：文稿划分的逻辑模块
           - ### 三级标题：该模块下的具体子观点
           - - 列表项：该观点对应的【原文核心金句/关键细节】（直接引用，不要改动太多）
        
        2. 编写准则：
           - 忠于原文：不要添加文稿中没提到的外部信息。
           - 拒绝空洞：不要只给小标题，必须在子观点下挂载原文中的核心论据或金句。
           - 清晰易读：金句部分请用“「 」”包裹。
        
        文稿内容：
        {result['text']}
        """
        
        response = client.chat.completions.create(model="qwen-plus", messages=[{"role": "user", "content": prompt}])
        
        map_name = "2_深度逻辑大纲.md"
        with open(map_name, "w", encoding="utf-8") as f:
            f.write(response.choices[0].message.content)
        
        files.download(map_name)
        print("🎉 深度解析完成！")

    except Exception as e:
        print(f"❌ 运行报错: {e}")

if __name__ == "__main__":
    run_podcast_tool()
