"""Serve the play page on its Hugging Face Space, which runs on the CPU."""

from __future__ import annotations

from shared_cache_world_model.play import PlaySettings, main

main(PlaySettings(device="cpu", open_browser=False))
