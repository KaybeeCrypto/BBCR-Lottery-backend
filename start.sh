#!/usr/bin/env bash
uvicorn Lottery_backend.app.main:app --host 0.0.0.0 --port $PORT

