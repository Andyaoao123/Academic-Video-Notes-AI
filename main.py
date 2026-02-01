import os, whisper, yt_dlp
from openai import OpenAI
from google.colab import files
import markdown # 用于渲染Markdown
from IPython.display import display, HTML # 用于在Colab显示HTML

# --- 外部接口区：变量放外面，README 脚本才改得到 ---
API_KEY = "sk-placeholder"
VIDEO_URL = "url-placeholder"

def run_podcast_tool():
    """
    核心执行函数。现在加入了：
    1. 语义纠错功能
    2. Colab 内置思维导图渲染
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
        
        # --- 新增：原始文稿的语义纠错与格式化 ---
        print("🔍 正在进行语义纠错与格式化...")
        raw_text_for_llm = f"""
        你是一位专业的文本校对专家和播客内容整理者。请对以下转录文稿进行两次处理：
        
        第一步：【语义纠错】
        仔细通读文稿，结合上下文逻辑，纠正所有明显的同音错别字、不通顺的语句，尤其是专有名词和关键概念（例如将“邻气”修正为“灵性”，“文芒”修正为“文盲”）。不要改变原始语义，但要让文稿更流畅、更准确。
        
        第二步：【段落格式化】
        将修正后的文稿重新分段，每段文字流畅且聚焦一个主题，去掉冗余的换行符，但保留原有的逻辑结构，使其更易阅读。
        
        请直接输出最终的【修正且格式化后的文稿】，不要包含任何额外说明或分析。

        原始转录文稿：
        {result['text']}
        """
        
        llm_client_corrector = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        correction_response = llm_client_corrector.chat.completions.create(
            model="qwen-plus", 
            messages=[{"role": "user", "content": raw_text_for_llm}]
        )
        corrected_text = correction_response.choices[0].message.content
        
        # 导出 1: 修正后的全文稿
        raw_name = "1_修正全文稿.txt"
        with open(raw_name, "w", encoding="utf-8") as f:
            f.write(corrected_text)
        files.download(raw_name)

        # C. 深度逻辑框架提取 (基于纠错后的文稿)
        print("✍️ 通义千问正在剥离深度逻辑骨架（包含核心金句）...")
        llm_client_extractor = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        # --- 强化的 Prompt 逻辑 ---
        prompt_extractor = f"""
        你是一位顶级的逻辑分析师。请将提供的【修正后的转录文稿】整理成一份【深度思维导图大纲】。
        
        要求如下：
        1. 结构化层级：
           - # 一级标题：播客核心主题
           - ## 二级标题：文稿划分的逻辑模块
           - ### 三级标题：该模块下的具体子观点
           - - 列表项：该观点对应的【原文核心金句/关键细节】（直接引用修正后的文本，并用“「 」”包裹）。
        
        2. 编写准则：
           - 忠于修正后的原文：不要添加文稿中没提到的外部信息。
           - 拒绝空洞：不要只给小标题，必须在子观点下挂载原文中的核心论据或金句。
           - 清晰易读：金句部分请用“「 」”包裹。
        
        修正后的文稿内容：
        {corrected_text}
        """
        
        response_extractor = llm_client_extractor.chat.completions.create(
            model="qwen-plus", 
            messages=[{"role": "user", "content": prompt_extractor}]
        )
        map_content = response_extractor.choices[0].message.content
        
        # 导出 2: 深度逻辑大纲
        map_name = "2_深度逻辑大纲.md"
        with open(map_name, "w", encoding="utf-8") as f:
            f.write(map_content)
        files.download(map_name)
        
        print("🎉 深度解析完成！请检查浏览器下载提示，并查看下方的 Colab 渲染图。")

        # --- 新增：在 Colab 内部渲染思维导图 ---
        print("\n--- Colab 内置思维导图预览 ---")
        # 由于 Colab 不直接支持 Markmap 等复杂渲染，这里先简单渲染成 HTML 格式，
        # 如果需要交互式导图，依然推荐复制到 Markmap.js.org 或幕布
        html_content = markdown.markdown(map_content, extensions=['fenced_code', 'tables', 'attr_list'])
        
        # 简单的Markdown转HTML显示，不具备完整思维导图的交互性
        # 如果需要更专业的交互式导图，仍推荐复制到 Markmap 或幕布
        display(HTML(f"""
        <style>
            body {{ font-family: sans-serif; line-height: 1.6; color: #333; }}
            h1 {{ color: #1a0dab; border-bottom: 2px solid #eee; padding-bottom: 5px; }}
            h2 {{ color: #006621; margin-top: 20px; }}
            h3 {{ color: #003366; margin-top: 15px; }}
            ul {{ list-style-type: disc; margin-left: 20px; }}
            li {{ margin-bottom: 5px; }}
            p {{ margin-top: 10px; }}
        </style>
        <div style="padding: 20px; border: 1px solid #ccc; border-radius: 8px; background-color: #f9f9f9;">
            <h2>💡 生成的思维导图大纲 (Markdown渲染，非交互式)</h2>
            {html_content}
            <hr>
            <p><strong>提示：</strong>如需获得交互式思维导图体验，请复制上方文本到 <a href="https://markmap.js.org/repl" target="_blank">Markmap</a> 或 <a href="https://mubu.com" target="_blank">幕布</a>。</p>
        </div>
        """))

    except Exception as e:
        print(f"❌ 运行报错: {e}")
        # 如果是API Key错误，给用户更明确的提示
        if "AuthenticationError" in str(e) or "invalid api_key" in str(e).lower():
            print("❗ 检查你的 API Key 是否正确填写或已过期。")

if __name__ == "__main__":
    run_podcast_tool()
