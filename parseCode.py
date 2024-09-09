import os
import json
import logging
from music21 import converter, note, chord

# Added logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_midi_directory(directory_path, pitch_shift_interval=0):

    data = []  # Using a set to store unique notes

    for root, _, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            logger.info("Parsing MIDI file: %s", file_path)
            parsed_data = parse_midi(file_path)
            data.extend(parsed_data)
            print("Number of unique notes in file:", len(parsed_data))

    return data  # Convert set back to a sorted list


def parse_midi(file_path, pitch_shift_interval=0):

    parsed_data = {"notes": [], "durations": [], "dynamics": []}

    try:
        midi = converter.parse(file_path)

        for part in midi.parts:
            for element in part.recurse():
                if isinstance(element, note.Note):
                    parsed_data["notes"].append(str(element.pitch))
                    parsed_data["durations"].append(
                        element.duration.quarterLength)

                    if hasattr(element, "volume"):
                        parsed_data["dynamics"].append(element.volume.velocity)
                    else:
                        parsed_data["dynamics"].append(None)
                elif isinstance(element, chord.Chord):
                    # Handle each note in the chord
                    for chord_note in element:
                        parsed_data["notes"].append(str(chord_note.pitch))
                        parsed_data["durations"].append(
                            element.duration.quarterLength)

                        if hasattr(element, "volume"):
                            parsed_data["dynamics"].append(
                                element.volume.velocity)
                        else:
                            parsed_data["dynamics"].append(None)
    except Exception as e:
        logger.error("Error parsing %s: %s", file_path, e)

    return parsed_data


def save_data_to_file(notes, output_file_path):
    with open(output_file_path, "w") as file:
        json.dump(notes, file, indent=2)


# Replace 'your_dataset_path' with the path to your MIDI dataset
dataset_path = r"C:\\Users\\ASUS\\Desktop\\AI\\Advance AI Music\\Composer data"
output_file_path = os.path.join(
    r'C:\Users\ASUS\Desktop\AI\Advance AI Music', 'parsed_notes.json')

# Use the functions
data = parse_midi_directory(dataset_path)

# Save the parsed notes to a file in JSON format
# This line is where the change was made
save_data_to_file(data, output_file_path)
