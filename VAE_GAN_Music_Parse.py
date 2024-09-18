import os
import json
import logging
from music21 import converter, note, chord, environment
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='music21')

# Added logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_midi_directory(directory_path, pitch_shift_interval=0):
    notes_with_details = []  # Using a list to store dictionaries with note details

    for root, _, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            logger.info("Parsing MIDI file: %s", file_path)
            parsed_notes = parse_midi(file_path)
            notes_with_details.extend(parsed_notes)
            print("Number of unique notes in file:", len(parsed_notes))

    return notes_with_details


def parse_midi(file_path, pitch_shift_interval=0):
    notes_with_details = []

    try:
        midi = converter.parse(file_path)

        for part in midi.parts:
            for element in part.recurse():
                if isinstance(element, note.Note):
                    note_details = {
                        "pitch": str(element.pitch),
                        "offset": float(element.offset),
                        "duration": float(element.duration.quarterLength),
                        "dynamic": element.volume.velocity if element.volume else None,
                    }
                    notes_with_details.append(note_details)
                elif isinstance(element, chord.Chord):
                    chord_details = {
                        "pitches": ['.'.join(str(n) for n in element.pitches)],
                        "offset": float(element.offset),
                        "duration": float(element.duration.quarterLength),
                        "dynamic": element.volume.velocity if element.volume else None,
                    }
                    notes_with_details.append(chord_details)
    except Exception as e:
        logger.error("Error parsing %s: %s", file_path, e)

    return notes_with_details


def save_notes_to_file(notes, output_file_path):
    try:
        with open(output_file_path, "w") as file:
            json.dump(notes, file, indent=2)
    except Exception as e:
        logger.error("Error saving notes to file: %s", e)


# Replace 'your_dataset_path' with the path to your MIDI dataset
dataset_path = "Song category"
output_file_path = os.path.join('parsed_notes.json')

# Use the functions
notes = parse_midi_directory(dataset_path)

# Save the parsed notes to a file in JSON format
# This line is where the change was made
save_notes_to_file(notes, output_file_path)
