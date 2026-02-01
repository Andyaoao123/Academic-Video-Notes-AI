import os, whisper, yt_dlp
from openai import OpenAI
from google.colab import files
import markdown
from IPython.display import display, HTML

# --- 外部接口区 ---
API_KEY = "sk-placeholder"
VIDEO_URL = "url-placeholder"

def run_podcast_tool():
    if "placeholder" in API_KEY or "placeholder" in VIDEO_URL:
        print("❌ 错误：请先在 Colab 中设置 main.API_KEY 和 main.VIDEO_URL")
        return

    try:
        # 1. 下载音频
        print(f"📥 正在抓取音频...")
        if os.path.exists("temp_audio.m4a"): os.remove("temp_audio.m4a")
        
        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'outtmpl': 'temp_audio.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([VIDEO_URL])
        
        # 2. 转录
        print("🎙️ Whisper 正在精准听写...")
        model = whisper.load_model("base") 
        result = model.transcribe("temp_audio.m4a")
        
        # 3. 语义整理 (全能通用纠错逻辑)
        print("🔍 正在进行语义整形 (自动纠错、断句、去废词)...")
        correction_prompt = f"""
        你是一位精通中文语境的资深编辑。请对以下杂乱的语音转文字稿进行“深度重塑”：
        
        要求：
        1. 【加标点与断句】：根据语义逻辑添加精准的标点符号。严禁出现超过15字而没有标点的“长龙句”，确保人类阅读节奏感。
        2. 【全能纠错】：结合全篇上下文，自动修正所有由于发音相似导致的识别错误（包括但不限于专有名词、医学术语、口语谐音）。
        3. 【去除口语赘词】：彻底过滤掉“那个、就是、然后、呃、我的话、对吧、所谓的”等毫无意义的填充词。
        4. 【分段处理】：每段只聚焦一个核心语义点，逻辑切换时必须换行。
        
        原始稿件内容：
        {result['text']}
        """
        
        client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        corr_res = client.chat.completions.create(
            model="qwen-plus", 
            messages=[{"role": "user", "content": correction_prompt}]
        )
        corrected_text = corr_res.choices[0].message.content

        # 4. 提取深度逻辑大纲
        print("✍️ 正在生成深度思维导图大纲...")
        map_prompt = f"""
        你是一位顶级的逻辑分析师。请基于以下【整理后的精排文稿】提取深度逻辑大纲。
        要求：
        - # 一级标题：播客核心主题
        - ## 二级标题：核心逻辑板块
        - ### 三级标题：具体论点/子观点
        - - 列表项：用「」包裹的原文核心金句。
        
        精排文稿内容：
        {corrected_text}
        """
        map_res = client.chat.completions.create(
            model="qwen-plus", 
            messages=[{"role": "user", "content": map_prompt}]
        )
        map_content = map_res.choices[0].message.content

        # 5. 下载文件
        with open("1_修正精排文稿.txt", "w", encoding="utf-8") as f: f.write(corrected_text)
        with open("2_深度逻辑大纲.md", "w", encoding="utf-8") as f: f.write(map_content)
        files.download("1_修正精排文稿.txt")
        files.download("2_深度逻辑大纲.md")

        # 6. Colab 预览渲染
        print("\n--- 📝 逻辑大纲实时预览 ---")
        display(HTML(f"""
        <div style="background:#f9f9f9; padding:20px; border-radius:12px; border:1px solid #ddd; line-height:1.8;">
            {markdown.markdown(map_content)}
        </div>
        """))
        print("\n🎉 任务完成！两个文件已自动下载。")

    except Exception as e:
        print(f"❌ 运行报错: {e}")

if __name__ == "__main__":
    run_podcast_tool()
