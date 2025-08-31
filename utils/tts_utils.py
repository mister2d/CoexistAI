
import logging
import random
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf
import asyncio

from kokoro_onnx import Kokoro

def random_pause(min_duration=0.5, max_duration=2.0, sample_rate=None):
    """
    Generate a random pause (silence) audio segment.
    """
    silence_duration = random.uniform(min_duration, max_duration)
    silence = np.zeros(int(silence_duration * sample_rate))
    return silence

import re

async def parse_podcast(text: str, voice_choices:list) -> list[dict]:
    """
    Parse a <podcast>...</podcast> transcript into list of {voice, text} dicts.
    Works even if speakers are in the same line.
    """

    # Identify unique speakers in the transcript
    match = re.search(r"<podcast>(.*?)</podcast>", text, re.DOTALL)
    if not match:
        return []
    content = match.group(1).strip()
    speakers = set(re.findall(r"\[(.*?)\]", content))

    # Map speakers to random voices from the given list
    def assign_voices(speaker_list, voice_choices):
        assigned = {}
        choices = voice_choices.copy()
        random.shuffle(choices)
        for i, speaker in enumerate(speaker_list):
            assigned[speaker] = choices[i % len(choices)]
        return assigned

    # Example: voice_choices = ["voice1", "voice2", "voice3"]
    voice_map = assign_voices(speakers, voice_choices)

    # Extract content inside <podcast> ... </podcast>
    match = re.search(r"<podcast>(.*?)</podcast>", text, re.DOTALL)
    if not match:
        return []

    content = match.group(1).strip()

    # Regex: find [Speaker] text until next [ or end
    segments = re.findall(r"\[(.*?)\]\s*([^[]+)", content)

    result = []
    for speaker, speech in segments:
        speaker = speaker.strip()
        speech = speech.strip()

        voice = voice_map.get(speaker, f"default_{speaker.lower()}")
        result.append({"voice": voice, "text": speech})
    logger.info(f"Parsed podcast segments: {result}")
    return result


async def podcasting(sentences, filename):
    """
    Generate a podcast audio file from the given sentences.
    """
    try:
        kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        logger.info("Initialized Kokoro TTS engine.")
        audio = []
        for sentence in sentences:
            voice = sentence["voice"]
            text = sentence["text"]
            logger.info(f"Creating audio with {voice}: {text}")
            samples, sample_rate = kokoro.create(
                text,
                voice=voice,
                speed=1.0,
                lang="en-us",
            )
            audio.append(samples)
            # Add random silence after each sentence
            audio.append(random_pause(sample_rate=sample_rate))

        # Concatenate all audio parts
        audio = np.concatenate(audio)
        
        # Save the generated audio to file
        sf.write(f"{filename}", audio, sample_rate)
        await asyncio.sleep(5)
        logger.info(f"Created {filename}")
    except Exception as e:
        logger.error(f"Error occurred while creating podcast: {e}")