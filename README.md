# Academic-Video-Notes-AI
一键式讲座转笔记工具：整合 yt-dlp, Whisper 与 Gemini API

🎓 Academic-Video-Notes-AI (学术讲座一键收割机)
🌟 项目简介
  这是一个专为准研究生/科研人打造的自动化工具。它能帮你把看不过来的 B 站、YouTube 学术视频，一键转化成排版精美、逻辑清晰的 Markdown 笔记。

为什么要做这个？
  导师发的讲座视频太长，没时间反复看。
  语音转文字乱码太多，没有重点。
  心理建设：把繁琐的整理工作交给 AI，让我们专注于思考。

🛠️ 核心功能
一键抓取：只需提供视频 URL，自动提取音频。
高精转录：使用 OpenAI 的 Whisper 模型进行本地识别。
学术排版：调用 Gemini 1.5 Pro 自动分段、修正专业术语、提取核心观点。
思维导图：自动生成可直接渲染的 Mermaid 思维导图代码。

🚀 SOP 快速开始 (Colab 版)
准备 Key：在 Google AI Studio 获取你的免费 API Key。
上传代码：将项目中的 main.py 上传至 Google Colab。
填入参数：
  在代码顶部找到 VIDEO_URL，粘贴视频地址。
  在 MY_API_KEY 位置填入你的密钥。
点火运行：点击运行，等待 .md 笔记自动弹出并下载。

⚠️ 避坑指南 (重要！必看)
在开发和使用过程中，我们总结了以下血泪教训：
引号地狱 (NameError)：
  错误示例：MY_KEY = AIzaSy... (报错：NameError)
  正确示例：MY_KEY = "AIzaSy..." (必须使用双引号包裹！)
门牌号错误 (404 Not Found)：
  如果报错 404，通常是接口地址版本不对。请尝试在 URL 中将 v1beta 切换为 v1。
视频源问题：
  由于 yt-dlp 更新很快，如果下载失败，请在 Colab 运行 !pip install -U yt-dlp 更新工具。
📝 开发随笔
  这个项目是我从“消费者”进化为“开发者”的第一个里程碑。它让我明白：技术不仅是拿来用的，更是拿来拆解和重塑的。 就算复刻的过程中满地报错，只要路通了，那就是胜利。
💡 怎么修改它？
  如果你觉得这个 README 写的不错，点击文件上方的 Edit (笔状图标)，把上面的文字直接粘进去，然后点击底部的 Commit changes 即可！
