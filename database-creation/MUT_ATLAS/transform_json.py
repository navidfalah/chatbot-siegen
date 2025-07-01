import json


def transform_location(location: dict) -> dict:
    """
    Transforms a single location object into a document for embedding and a metadata payload.

    Args:
        location: A dictionary representing one location.

    Returns:
        A dictionary containing the 'document_to_embed' string and the 'payload' dictionary.
    """
    # 1. Group the multifield values for easier access
    multifields = {}
    for item in location.get("mfAngebotMultifieldValues", []):
        field_name = item.get("fmMultiField", {}).get("name")
        field_value = item.get("fmMultiFieldValue", {}).get("value")
        if field_name and field_value:
            # Clean up the key name for better readability
            clean_key = field_name.replace(
                "Kat_", "").replace("_", " ").lower()
            if clean_key not in multifields:
                multifields[clean_key] = []
            multifields[clean_key].append(field_value)

    # 2. Create the descriptive text document for embedding
    doc_parts = []
    if location.get("name"):
        doc_parts.append(f"The service is called {location['name']}.")
    if location.get("katWasSum"):
        doc_parts.append(f"It offers: {location['katWasSum']}.")
    if location.get("art"):
        doc_parts.append(f"It is a {location['art']} type of service.")
    if location.get("ort"):
        doc_parts.append(f"It is located in {location['ort']}.")

    # Add details from the multifields to the document
    if 'themen' in multifields:
        doc_parts.append(
            f"Main topics include: {', '.join(multifields['themen'])}.")
    if 'zielgruppe' in multifields:
        doc_parts.append(
            f"The target audience is: {', '.join(multifields['zielgruppe'])}.")
    if 'wasdetail' in multifields:
        doc_parts.append(
            f"Specific offerings are: {', '.join(multifields['wasdetail'])}.")

    document_to_embed = " ".join(doc_parts)

    # 3. Create the structured metadata payload
    payload = {
        "id": location.get("id"),
        "name": location.get("name"),
        "contact": {
            "email": location.get("email") or None,
            "phone": location.get("telefon") or None,
            "homepage": location.get("homePage") or None
        },
        "address": {
            "street": location.get("strasse") or None,
            "house_number": location.get("hausnummer") or None,
            "zip_code": location.get("plz") or None,
            "city": location.get("ort") or None
        },
        "location_geo": {
            "lat": location.get("latitude"),
            "lon": location.get("longitude")
        } if location.get("latitude") and location.get("longitude") else None,
        "details": multifields
    }

    return {
        "document_to_embed": document_to_embed,
        "payload": payload
    }


def process_json_file(input_filepath: str, output_filepath: str):
    """
    Reads a large JSON file with locations, transforms them, and saves to a new file.

    Args:
        input_filepath: Path to the source JSON file.
        output_filepath: Path to save the transformed data.
    """
    print(f"Starting transformation of '{input_filepath}'...")

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{input_filepath}' is not a valid JSON file.")
        return

    # Assuming the root object has a key "locations" which is a list
    locations_list = data.get("locations")
    if locations_list is None:
        print("Error: JSON file must have a root key 'locations' containing a list of objects.")
        return

    # Transform each location in the list
    transformed_data = [transform_location(loc) for loc in locations_list]

    # Save the transformed data to the output file
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully transformed {len(transformed_data)} locations.")
    print(f"Transformed data saved to '{output_filepath}'.")


# --- HOW TO USE ---
# 1. Save this code as a Python file (e.g., `transform.py`).
# 2. Place your large JSON file (e.g., `all_locations.json`) in the same directory.
# 3. Update the file paths in the line below.
# 4. Run the script from your terminal: python transform.py

if __name__ == "__main__":
    # <-- CHANGE TO YOUR INPUT FILENAME
    INPUT_FILE = "mut_atlas_data_20250620_163050.json"
    OUTPUT_FILE = "transformed_for_qdrant.json"  # <-- NAME FOR THE OUTPUT FILE

    process_json_file(INPUT_FILE, OUTPUT_FILE)
