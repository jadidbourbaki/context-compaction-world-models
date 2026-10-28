---
title: Context Compaction for World Models
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
---

A trained world model plays a generated game of walled rooms. The agent observes a room on
its first visit and asks the model to remember the room on every revisit. The loop buttons
send the agent on a walk that returns to its start room after 4 to 128 steps, so each loop
ends in a revisit that tests the model's memory. The page compares four architectures and
two ways of building the model's context.

The weights come from
[jadidbourbaki/context-compaction-world-models](https://huggingface.co/jadidbourbaki/context-compaction-world-models).
