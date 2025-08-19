from diarize import diarize
from transcribe import transcribe
from clean import clean_transcription
import subprocess
import os

TEMP_FILE = "temp_audio.wav"


def extract_audio(input_file, output_audio_file=TEMP_FILE):
    """
    Converts input file (mp4, m4a, or mp3) to wav format using ffmpeg.
    """
    subprocess.run(
        ["ffmpeg", "-i", input_file, "-q:a", "0", "-map", "a", output_audio_file, "-y"],
        stdout=subprocess.DEVNULL,
        check=True,
    )
    return output_audio_file


if __name__ == "__main__":
    AUDIO_DIR = "./audio"
    OUTPUT_DIR = "./output"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Iterate over all files in the directory
    for file_name in os.listdir(AUDIO_DIR):
        input_file_path = os.path.join(AUDIO_DIR, file_name)

        # Skip unsupported file types
        if not file_name.endswith((".mp4", ".wav", ".m4a", ".mp3", ".mov")):
            print(f"Skipping unsupported file type: {file_name}")
            continue

        try:
            # Step 0: Extract audio if necessary
            if input_file_path.endswith((".mp4", ".m4a", ".mp3", ".mov")):
                input_file_path = extract_audio(input_file_path)

            # Step 1: Diarize
            diarization_output_file_path = os.path.join(OUTPUT_DIR, file_name + ".json")
            print(f"Diarizing {file_name}...")
            diarize(input_file_path, diarization_output_file_path)

            # Step 2: Transcribe
            transcribe_output_file_path = os.path.join(OUTPUT_DIR, file_name + ".txt")
            print(f"Transcribing {file_name}...")
            transcribe(
                input_file_path,
                diarization_output_file_path,
                transcribe_output_file_path,
            )

            # Step 3: Clean
            final_output_file_path = os.path.join(
                OUTPUT_DIR, file_name + "_cleaned.txt"
            )
            print(f"Cleaning transcription for {file_name}...")
            clean_transcription(transcribe_output_file_path, final_output_file_path)

        finally:
            # Cleanup temporary file
            if os.path.exists(TEMP_FILE):
                os.remove(TEMP_FILE)
                print(f"Temporary file {TEMP_FILE} removed.")

        print(f"Processing completed for {file_name}.\n")
