def clean_transcription(input_file_path, output_file_path):
    with open(input_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    processed_lines = []
    previous_speaker = None
    current_text = ""

    for line in lines:
        line = line.strip()
        if not line:  # Skip blank lines
            continue

        parts = line.split(" | ")
        if len(parts) < 3:  # Skip lines that don't match the expected format
            continue

        speaker, timestamp, text = parts[0], parts[1], " | ".join(parts[2:])

        if speaker == previous_speaker:
            # Append text if the same speaker as previous line
            current_text += " " + text
        else:
            if previous_speaker is not None:
                # Save the previous speaker's text
                processed_lines.append(
                    f"{previous_speaker} | {start_timestamp} | {current_text}\n"
                )

            # Start a new line for a new speaker
            previous_speaker = speaker
            start_timestamp = timestamp
            current_text = text

    # Don't forget to add the last line
    if previous_speaker is not None:
        processed_lines.append(
            f"{previous_speaker} | {start_timestamp} | {current_text}\n"
        )

    # Write processed lines to a new file
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.writelines(processed_lines)
