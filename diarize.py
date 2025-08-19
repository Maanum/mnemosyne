import json
from pyannote.audio import Pipeline
from pydub import AudioSegment


def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000.0  # Duration in seconds


def diarize_speakers(file_path):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="",
    )
    diarization = pipeline(file_path)
    return diarization


def save_diarization_to_file(diarization, output_file_path):
    diarization_data = [
        {"start": segment.start, "end": segment.end, "speaker": speaker}
        for segment, _, speaker in diarization.itertracks(yield_label=True)
    ]
    try:
        with open(output_file_path, "w", encoding="utf-8") as file:
            json.dump(diarization_data, file, indent=4)
        print(f"Diarization saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving diarization: {e}")


def diarize(input_file_path, output_file_path):
    # Rest of the pipeline
    # audio_duration = get_audio_duration(input_file_path)
    diarization = diarize_speakers(input_file_path)
    save_diarization_to_file(diarization, output_file_path)

    return
