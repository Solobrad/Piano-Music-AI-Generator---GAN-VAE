import json


def check_empty_or_null(json_file):
    errors_found = False  # Flag to track if any errors are found

    # Load the JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Check each item in the JSON data
    for index, item in enumerate(data):
        pitch = item.get('pitch', None)
        pitches = item.get('pitches', None)

        # Check for empty or null 'pitch'
        if pitch is None or pitch == "":
            print(f"Item {index} has an empty or null 'pitch' field: {item}")
            errors_found = True

        # Check for empty or null values in 'pitches' list
        if pitches is not None:
            empty_pitches = [p for p in pitches if p is None or p == ""]
            if empty_pitches:
                print(
                    f"Item {index} has empty or null pitch(es) in 'pitches' list: {item}")
                errors_found = True
        elif pitch is None and pitches is None:
            print(f"Item {index} has neither 'pitch' nor 'pitches': {item}")
            errors_found = True

    if not errors_found:
        print("No errors found.")


# Replace 'parsed_notes.json' with the path to your JSON file
json_file_path = 'parsed_notes.json'
check_empty_or_null(json_file_path)
