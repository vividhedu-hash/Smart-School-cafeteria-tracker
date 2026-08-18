#!/usr/bin/env bash
# One-click Launcher for Smart Cafeteria Waste Tracker
cd "$(dirname "$0")"
if [ -d ".venv" ]; post
then
    .venv/bin/python run.py
else
    python3 run.py
fi
