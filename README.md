# 🚀 Academic-Video-Notes-AI (学术视频/播客收割机) 🎓

这是一个专为**深度学习者**设计的自动化工具。它能一键将 B站、播客或 YouTube 的长视频/音频转化为**带原文金句的思维导图逻辑框架**。



## ✨ 项目亮点
- 🎙️ **精准听写**：调用 OpenAI Whisper 模型，即使是嘈杂的播客也能准确转录。
- ✍️ **深度拆解**：不仅提供小标题，更自动剥离出**子观点**与**原文金句「」**。
- 🔗 **全能支持**：支持 Bilibili、小宇宙 FM、YouTube 等几乎所有主流平台。
- 📂 **即拿即用**：自动导出 `.txt` 全文和 `.md` 思维导图大纲。

---

🛠️ 工作流说明
音频抓取：使用 yt-dlp 提取最高音质音频。

语音转文字：Whisper (Base) 模型进行全文转录。

逻辑剥离：通过 qwen-plus (通义千问) 提取深度逻辑框架。

自动下载：任务完成后，浏览器会自动弹出两个下载任务。

📝 进阶技巧
生成的 .md 文件可以完美导入 幕布 (Mubu)、XMind 或 Obsidian。

建议将 .md 文件直接拖入 XMind，瞬间生成一张逻辑严密的知识地图。

🤝 贡献与反馈
如果你觉得这个工具有点意思，欢迎给个 Star ⭐！



## ⚡ 极速开始 (Colab 一键运行)

不需要安装任何环境，直接在 [Google Colab](https://colab.research.google.com/) 新建一个单元格，粘贴并运行以下代码：

```python
# 1. 环境准备 (每次重启会话需跑一次)
print("🛠️ 环境安装中...")
!pip install -q openai-whisper openai yt-dlp

# 2. 从 GitHub 抓取最新逻辑代码
print("📥 正在同步 GitHub 最新脚本...")
!curl -O [https://raw.githubusercontent.com/Andyaoao123/Academic-Video-Notes-AI/main/main.py](https://raw.githubusercontent.com/Andyaoao123/Academic-Video-Notes-AI/main/main.py)

# 3. 导入模块并刷新
import main, importlib
importlib.reload(main) 

# 4. 【核心配置】在此填入你的参数
main.API_KEY = "你的sk-key" 
main.VIDEO_URL = "[https://www.bilibili.com/video/BV16o6qBgETj/](https://www.bilibili.com/video/BV16o6qBgETj/)"

# 5. 启动自动化流程
print("🚀 启动自动化流程...")
main.run_podcast_tool()














