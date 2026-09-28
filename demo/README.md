---
title: AI Blogger Run Replay
emoji: 📝
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.24.0
app_file: app.py
pinned: false
---

# AI Blogger: run replay

Step by step replays of real, recorded runs of a local multi agent blogging pipeline: a
planner, parallel research branches that pull quoted claims from web pages, a Gemini
auditor that checks every claim against its quote, a cross check for gaps and
disagreements, then a writer and an editor. One recording is an attack topic, showing the
pipeline research the subject instead of obeying the injected command.

Nothing here calls a model or an API. Recordings live in `runs/` and are refreshed with
`python demo/record.py` in the main repository:
https://github.com/PrashantMaht0/multi_agent_ai_blogger
