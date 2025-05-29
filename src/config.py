APPLICATION_WIDTH = 85
THEME = "DarkGray12"
TRANSPARENCY = 0.65  # 0.0 is fully transparent, 1.0 is opaque

OUTPUT_FILE_NAME = "record.wav"
OUTPUT_TRANSCRIPT_FILE_NAME = "record.wav.txt"
SAMPLE_RATE = 48000

MODELS = ["gpt-4.1","gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
DEFAULT_MODEL = MODELS[0]

DEFAULT_POSITION = "Engineering Manager"

WHISPER_CLI_PATH: str = "~/Documents/source/interview-coder/whisper.cpp/build/bin/whisper-cli"
# base model
WHISPER_MODEL_PATH: str = "~/Documents/source/interview-coder/whisper.cpp/models/ggml-base.en.bin"
# medium model
WHISPER_MODEL_PATH_MEDIUM: str = "~/Documents/source/interview-coder/whisper.cpp/models/ggml-medium.en.bin"
