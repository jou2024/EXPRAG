import json


def process_points(data):
    """
    Processes points from the 'what_to_do' and 'warning_signs' sections of the input JSON data.

    The function:
    - Extracts points from the "what_to_do" and "warning_signs" categories.
    - Adds an index, summary, sentence, and category to each point.
    - Groups the points into sub-lists, each containing 3–5 points, following these rules:
        - If the total number of points modulo 4 is 0, group every 4 points together.
        - If the modulo is 1, group the last 5 points together.
        - If the modulo is 2, split the last 6 points into two groups of 3.
        - If the modulo is 3, group the last 3 points together.

    Parameters:
    - data (dict): A dictionary containing "points" with "what_to_do" and "warning_signs".

    Returns:
    - list: A list of grouped points, where each group contains 3–5 points.
    """
    # Extract points from "what_to_do" and "warning_signs"
    what_to_do_points = data["points"]["what_to_do"]
    warning_signs_points = data["points"]["warning_signs"]

    # Combine points and add index, summary, and sentence only
    all_points = []
    index = 1
    for point in what_to_do_points:
        all_points.append({
            "index": index,
            "summary": point["summary"],
            "sentence": point["sentence"],
            "category": "what_to_do"
        })
        index += 1

    for point in warning_signs_points:
        all_points.append({
            "index": index,
            "summary": point["summary"],
            "sentence": point["sentence"],
            "category": "warning_signs"
        })
        index += 1

    # Determine how to split the points based on len(all_points) % 4
    groups = []
    n = len(all_points)
    if n <=2:
        return False

    # First, group all except the last points based on n % 4
    i = 0
    while i < n:
        if n - i == 5 and n % 4 == 1:  # If there are exactly 5 points left
            groups.append(all_points[i:i + 5])
            break
        elif n - i == 6 and n % 4 == 2:  # If there are exactly 6 points left
            groups.append(all_points[i:i + 3])
            groups.append(all_points[i + 3:i + 6])
            break
        elif n - i == 3 and n % 4 == 3:  # If there are exactly 3 points left
            groups.append(all_points[i:i + 3])
            break
        else:  # Otherwise, group 4 points at a time
            groups.append(all_points[i:i + 4])
        i += 4

    return groups


# Example usage
json_data = '''{
  "reason": "The instruction includes detailed steps on what the patient should do regarding medications, wound care, activity, and weight bearing. However, it lacks specific warning signs that require contacting a doctor.",
  "evaluation": "Fail",
  "selected_content": [],
  "points": {
    "what_to_do": [
      {
        "summary": "Take medications as prescribed",
        "keywords": ["take medications", "prescribed medications"],
        "sentence": "Please take all medications as prescribed by your physicians at discharge."
      },
      {
        "summary": "Continue home medications",
        "keywords": ["continue medications", "home medications"],
        "sentence": "Continue all home medications unless specifically instructed to stop by your surgeon."
      },
      {
        "summary": "Avoid alcohol and machinery with narcotics",
        "keywords": ["avoid alcohol", "avoid machinery", "narcotic pain relievers"],
        "sentence": "Do not drink alcohol, drive a motor vehicle, or operate machinery while taking narcotic pain relievers."
      },
      {
        "summary": "Prevent constipation with water and stool softener",
        "keywords": ["prevent constipation", "drink water", "stool softener"],
        "sentence": "Narcotic pain relievers can cause constipation, so you should drink eight 8oz glasses of water daily and take a stool softener (colace) to prevent this side effect."
      },
      {
        "summary": "Take lovenox daily",
        "keywords": ["take lovenox", "anticoagulation"],
        "sentence": "Please take lovenox 40mg daily for 2 weeks."
      },
      {
        "summary": "Shower, no baths or swimming",
        "keywords": ["shower", "no baths", "no swimming"],
        "sentence": "You may shower. No baths or swimming for at least 4 weeks."
      },
      {
        "summary": "Stitches or staples removal",
        "keywords": ["stitches removal", "staples removal"],
        "sentence": "Any stitches or staples that need to be removed will be taken out at your 2-week follow up appointment."
      },
      {
        "summary": "No dressing if wound non-draining",
        "keywords": ["no dressing", "wound non-draining"],
        "sentence": "No dressing is needed if wound continues to be non-draining."
      },
      {
        "summary": "Keep splint on until follow-up",
        "keywords": ["keep splint on", "follow-up appointment"],
        "sentence": "Splint must be left on until follow up appointment unless otherwise instructed."
      },
      {
        "summary": "Do not get splint wet",
        "keywords": ["do not get splint wet", "keep splint dry"],
        "sentence": "Do NOT get splint wet."
      },
      {
        "summary": "Nonweight bearing on right lower extremity",
        "keywords": ["nonweight bearing", "right lower extremity"],
        "sentence": "Nonweight bearing in the right lower extremity."
      }
    ],
    "warning_signs": []
  }
}'''

if __name__ == "__main__":
    # Load the json input as a dictionary
    data = json.loads(json_data)

    # Process points and generate output
    output = process_points(data)

    # Output the result
    for group in output:
        print(json.dumps(group, indent=2))

    first_points_for_1_q = output[0]
    print("For the first question:")
    print(first_points_for_1_q)