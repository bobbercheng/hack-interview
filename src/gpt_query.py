from dotenv import load_dotenv
from loguru import logger
from openai import ChatCompletion, OpenAI
import subprocess
import os
from src.config import DEFAULT_MODEL, DEFAULT_POSITION, OUTPUT_FILE_NAME, WHISPER_CLI_PATH, WHISPER_STREAM_PATH, WHISPER_MODEL_PATH, LIVE_OUTPUT_TRANSCRIPT_FILE_NAME
import tiktoken

SYS_PREFIX: str = "You are interviewing for a "
SYS_SUFFIX: str = """ position.
You will receive an audio transcription of the question. It may not be complete. You need to understand the question and write an answer to it.\n
"""

SHORT_INSTRUCTION: str = "Concisely respond, limiting your answer to 50 words."
LONG_INSTRUCTION: str = "Before answering, take a deep breath and think one step at a time. Believe the answer in no more than 150 words."

load_dotenv()

client: OpenAI = OpenAI()

def transcribe_audio_with_whisper(path_to_file: str = OUTPUT_FILE_NAME) -> str:
    """
    Use whisper to transcribe the audio file. Transcribe cmd is 
    whisper.cpp/whisper-cli -m whisper.cpp/models/ggml-base.en.bin -f path_to_file --output-txt
    It will generate a transcription text file with the same name as the audio file, but with postfix .txt.
    After transcribing, read the transcription text file and return the transcription.
    If the transcription file exists, append the transcription to the file.
    """
    if os.path.exists(f"{path_to_file}.txt"):
        with open(f"{path_to_file}.txt", "r") as file:
            transcription: str = file.read()
        logger.debug(f"Transcription file exists: f{path_to_file}.txt")
    else:
        transcription: str = ""

    transcribe_cmd: str = f"{WHISPER_CLI_PATH} -m {WHISPER_MODEL_PATH} -f {path_to_file} --output-txt"
    transcribe_cmd_output: str = subprocess.run(transcribe_cmd, shell=True, capture_output=True, text=True)
    logger.debug(f"Transcribe cmd output: {transcribe_cmd_output}")
    with open(f"{path_to_file}.txt", "r") as file:
        transcription += file.read()
    logger.debug("Audio transcribed.")
    print("Transcription:", transcription)
    return transcription

# global variants
live_process: subprocess.Popen = None

def transcribe_live_audio_with_whisper(path_to_file: str = LIVE_OUTPUT_TRANSCRIPT_FILE_NAME):
    """
    Transcribe live audio with whisper.cpp.
    whisper.cpp/whisper-stream -m WHISPER_MODEL_PATH -t 6 --step 0 --length 30000 -vth 0.8 -c 0 -f {LIVE_OUTPUT_TRANSCRIPT_FILE_NAME}
    It will generate a transcription text file LIVE_OUTPUT_TRANSCRIPT_FILE_NAME.
    As it's live transcription, above command will run forever, we need to read transcript file to get the transcription 
    New transcription can be detected by output pipe change of the command. Ideally it should send a signal for new transcription.
    """
    global live_process
    if live_process is not None:
        return
    transcribe_cmd: str = f"{WHISPER_STREAM_PATH} -m {WHISPER_MODEL_PATH} -t 6 --step 0 --length 30000 -vth 0.8 -c 0 -f {LIVE_OUTPUT_TRANSCRIPT_FILE_NAME}"
    logger.debug(f"Running live transcribe with whisper: {transcribe_cmd}...")
    # run the command and print command output as a stream
    live_process = subprocess.Popen(transcribe_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True:
        try:
            output = live_process.stdout.readline()
            if output == b"" and live_process.stdout.at_eof():
                break
            if output:
                # logger.debug(output.decode(), end="")
                pass
        except Exception as error:
            logger.error(f"Can't read out of live transcribe: {error}")
            break
    if live_process is not None:
        live_process.wait()

def stop_transcribe_live():
    global live_process
    if live_process is None:
        return
    # Terminate process
    logger.debug(f"Terminating live transcribe with whisper...")
    live_process.terminate()
    live_process = None

def transcribe_audio_with_openai(path_to_file: str = OUTPUT_FILE_NAME) -> str:
    """
    Transcribe audio from a file using the OpenAI Whisper API.

    Args:
        path_to_file (str, optional): Path to the audio file. Defaults to OUTPUT_FILE_NAME.

    Returns:
        str: The audio transcription.
    """
    logger.debug(f"Transcribing audio from: {path_to_file}...")

    with open(path_to_file, "rb") as audio_file:
        try:
            transcript: str = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="text"
            )
        except Exception as error:
            logger.error(f"Can't transcribe audio: {error}")
            raise error

    logger.debug("Audio transcribed.")
    print("Transcription:", transcript)

    return transcript

def transcribe_audio(path_to_file: str = OUTPUT_FILE_NAME) -> str:
    return transcribe_audio_with_whisper(path_to_file)

def truncate_to_fit_context(
    text: str,
    context_window: int = 1_047_576,
    reserved_tokens: int = 200,
    model: str = "gpt-4.1"
) -> str:
    """
    Truncate `text` so that its token length ≤ context_window - reserved_tokens.

    Args:
        text:           The input string to truncate.
        context_window: Total context window size of the model.
        reserved_tokens: Number of tokens to reserve (e.g. for system prompts).
        model:          The OpenAI model name to choose the token encoding.

    Returns:
        A truncated string whose token count fits within the available window.
    """
    # load the encoding for the specified model (falls back to cl100k_base)
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    max_user_tokens = context_window - reserved_tokens
    tokens = enc.encode(text)

    if len(tokens) <= max_user_tokens:
        return text

    # keep only the first max_user_tokens tokens
    truncated = tokens[:max_user_tokens]
    return enc.decode(truncated)

def generate_answer(
    transcript: str,
    short_answer: bool = True,
    temperature: float = 0.7,
    model: str = DEFAULT_MODEL,
    position: str = DEFAULT_POSITION,
) -> str:
    """
    Generate an answer to the question using the OpenAI API.

    Args:
        transcript (str): The audio transcription.
        short_answer (bool, optional): Whether to generate a short answer. Defaults to True.
        temperature (float, optional): The temperature to use. Defaults to 0.7.
        model (str, optional): The model to use. Defaults to DEFAULT_MODEL.
        position (str, optional): The position to use. Defaults to DEFAULT_POSITION.

    Returns:
        str: The generated answer.
    """
    # truncate transcript to 10000 characters
    safe_transcript = truncate_to_fit_context(transcript)
    # Generate system prompt
    system_prompt: str = SYS_PREFIX + position + SYS_SUFFIX
    if short_answer:
        system_prompt += SHORT_INSTRUCTION
    else:
        system_prompt += LONG_INSTRUCTION

    # Generate answer
    try:
        response: ChatCompletion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": safe_transcript},
            ],
        )
    except Exception as error:
        logger.error(f"Can't generate answer: {error}")
        raise error

    return response.choices[0].message.content
