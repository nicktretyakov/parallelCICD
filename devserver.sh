#!/bin/sh
source .venv/bin/activate
python -m flask --app main run --debug
python ml_ci_cd_python/main.py --config ml_ci_cd_python/pipeline_config.yaml