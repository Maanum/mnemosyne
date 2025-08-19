import json
from pydub import AudioSegment
import whisper
import numpy as np
from tqdm import tqdm


def load_diarization_from_file(input_file_path):
    with open(input_file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def format_timestamp(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def transcribe_audio_segment(model, segment_audio):

    if segment_audio.frame_rate != 16000:  # 16 kHz
        segment_audio = segment_audio.set_frame_rate(16000)
    if segment_audio.sample_width != 2:  # int16
        segment_audio = segment_audio.set_sample_width(2)
    if segment_audio.channels != 1:  # mono
        segment_audio = segment_audio.set_channels(1)
    arr = np.array(segment_audio.get_array_of_samples())
    arr = arr.astype(np.float32) / 32768.0

    result = model.transcribe(arr, language="en")
    return result["text"]


def transcribe(
    audio_file_path, diarization_file_path, output_file_path, max_segments=None
):
    max_segments_to_process = None  # Set this to None to process all segments
    # Load the Whisper model
    model = whisper.load_model("medium")

    # Load the diarization data
    diarization_data = load_diarization_from_file(diarization_file_path)

    # Load the audio file
    audio = AudioSegment.from_file(audio_file_path)

    # Determine the number of segments to process
    total_segments = (
        len(diarization_data)
        if max_segments is None
        else min(max_segments, len(diarization_data))
    )
    # Open the output file
    with open(output_file_path, "w", encoding="utf-8") as output_file, tqdm(
        total=total_segments, desc="Transcribing", unit="segment"
    ) as pbar:
        for segment in diarization_data[:total_segments]:
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            speaker = segment["speaker"]

            # Extract the audio segment
            segment_audio = audio[start_ms:end_ms]

            # Transcribe the segment
            transcription = transcribe_audio_segment(model, segment_audio)

            # Format and write to file
            start_time = format_timestamp(segment["start"])
            output_file.write(f"{speaker} | {start_time} | {transcription}\n")

            # Update the progress bar
            pbar.update(1)
    return
