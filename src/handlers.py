from typing import Any, Dict

import FreeSimpleGUI as sg
from loguru import logger

from src import audio, gpt_query
from src.button import OFF_IMAGE, ON_IMAGE
import os
from src.config import OUTPUT_TRANSCRIPT_FILE_NAME, OUTPUT_FILE_NAME, LIVE_OUTPUT_TRANSCRIPT_FILE_NAME
import time

def handle_events(window: sg.Window, event: str, values: Dict[str, Any]) -> None:
    """
    Handle the events. Record audio, transcribe audio, generate quick and full answers.

    Args:
        window (sg.Window): The window element.
        event (str): The event.
        values (Dict[str, Any]): The values of the window.
    """
    # If the user is not focused on the position input, process the events
    focused_element: sg.Element = window.find_element_with_focus()
    if not focused_element or focused_element.Key != "-POSITION_INPUT-":
        if event in ("r", "R", "-RECORD_BUTTON-"):
            recording_event(window)
        elif event in ("a", "A", "-ANALYZE_BUTTON-"):
            transcribe_event(window)
        elif event in ("c", "C", "-CLEAR_BUTTON-"):
            clear_history(window)

    # If the user is focused on the position input
    if event[:6] in ("Return", "Escape"):
        window["-ANALYZE_BUTTON-"].set_focus()

    # When the transcription is ready
    elif event == "-WHISPER-":
        answer_events(window, values)

    # When the quick answer is ready
    elif event == "-QUICK_ANSWER-":
        logger.debug("Quick answer generated.")
        print("Quick answer:", values["-QUICK_ANSWER-"])
        window["-QUICK_ANSWER-"].update(values["-QUICK_ANSWER-"])

    # When the full answer is ready
    elif event == "-FULL_ANSWER-":
        logger.debug("Full answer generated.")
        print("Full answer:", values["-FULL_ANSWER-"])
        window["-FULL_ANSWER-"].update(values["-FULL_ANSWER-"])


def recording_event(window: sg.Window) -> None:
    """
    Handle the recording event. Record audio and update the record button.

    Args:
        window (sg.Window): The window element.
    """
    button: sg.Element = window["-RECORD_BUTTON-"]
    button.metadata.state = not button.metadata.state
    button.update(image_data=ON_IMAGE if button.metadata.state else OFF_IMAGE)

    # Record audio
    if button.metadata.state:
        window.perform_long_operation(lambda: audio.record(button), "-RECORDED-")
        window.perform_long_operation(gpt_query.transcribe_live_audio_with_whisper, "-RECORDED-")
        
        # Start periodic update of transcript text
        def update_transcript():
            last_content = ""
            while button.metadata.state:
                try:
                    if os.path.exists(LIVE_OUTPUT_TRANSCRIPT_FILE_NAME):
                        with open(LIVE_OUTPUT_TRANSCRIPT_FILE_NAME, 'r') as f:
                            transcript = f.read()
                            if transcript != last_content:
                                logger.debug(f"Transcript: {transcript}")
                                window["-TRANSCRIBED_TEXT-"].update(transcript, append=False)
                                window["-TRANSCRIBED_TEXT-"].set_vscroll_position(1.0)  # Scroll to bottom
                                last_content = transcript
                except Exception as e:
                    logger.error(f"Error reading transcript file: {e}")
                time.sleep(0.1)
        
        window.perform_long_operation(update_transcript, "-UPDATE_TRANSCRIPT-")
    else:
        gpt_query.stop_transcribe_live()


def transcribe_event(window: sg.Window) -> None:
    """
    Handle the transcribe event. Transcribe audio and update the text area.

    Args:
        window (sg.Window): The window element.
    """
    transcribed_text: sg.Element = window["-TRANSCRIBED_TEXT-"]
    transcribed_text.update("Transcribing audio...")

    # Transcribe audio
    # window.perform_long_operation(gpt_query.transcribe_audio, "-WHISPER-")
    def get_live_transcript():
        with open(LIVE_OUTPUT_TRANSCRIPT_FILE_NAME, 'r') as f:
            transcript = f.read()
            return transcript
    window.perform_long_operation(get_live_transcript, "-WHISPER-")


def answer_events(window: sg.Window, values: Dict[str, Any]) -> None:
    """
    Handle the answer events. Generate quick and full answers and update the text areas.

    Args:
        window (sg.Window): The window element.
        values (Dict[str, Any]): The values of the window.
    """
    transcribed_text: sg.Element = window["-TRANSCRIBED_TEXT-"]
    quick_answer: sg.Element = window["-QUICK_ANSWER-"]
    full_answer: sg.Element = window["-FULL_ANSWER-"]

    # Get audio transcript and update text area
    audio_transcript: str = values["-WHISPER-"]
    transcribed_text.update(audio_transcript)

    # Get model and position
    model: str = values["-MODEL_COMBO-"]
    position: str = values["-POSITION_INPUT-"]

    # Generate quick answer
    logger.debug("Generating quick answer...")
    quick_answer.update("Generating quick answer...")
    window.perform_long_operation(
        lambda: gpt_query.generate_answer(
            audio_transcript,
            short_answer=True,
            temperature=0,
            model=model,
            position=position,
        ),
        "-QUICK_ANSWER-",
    )

    # Generate full answer
    logger.debug("Generating full answer...")
    full_answer.update("Generating full answer...")
    window.perform_long_operation(
        lambda: gpt_query.generate_answer(
            audio_transcript,
            short_answer=False,
            temperature=0.7,
            model=model,
            position=position,
        ),
        "-FULL_ANSWER-",
    )

def clear_history(window: sg.Window) -> None:
    """
    Clear the history.

    Args:
        window (sg.Window): The window element.
    """
    logger.debug("Clearing history...")
    # delete OUTPUT_TRANSCRIPT_FILE_NAME and OUTPUT_FILE_NAME
    if os.path.exists(OUTPUT_TRANSCRIPT_FILE_NAME):
        os.remove(OUTPUT_TRANSCRIPT_FILE_NAME)
    if os.path.exists(OUTPUT_FILE_NAME):
        os.remove(OUTPUT_FILE_NAME)
    if os.path.exists(LIVE_OUTPUT_TRANSCRIPT_FILE_NAME):
        os.remove(LIVE_OUTPUT_TRANSCRIPT_FILE_NAME)
    window["-TRANSCRIBED_TEXT-"].update("")
    window["-QUICK_ANSWER-"].update("")
    window["-FULL_ANSWER-"].update("")
    logger.debug(f"History cleared. {OUTPUT_TRANSCRIPT_FILE_NAME} and {OUTPUT_FILE_NAME} deleted.")
