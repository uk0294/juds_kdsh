import csv

def save_results_to_csv(results, output_filename):
    """Save results to a CSV file."""
    headers = ["Paper ID", "Publishable", "Conference", "Rationale"]
    with open(output_filename, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for result in results:
            paper_id = result.get("Paper ID", "N/A")
            classification = result.get("Classification", "Conference: NA")
            rationale = result.get("Reasoning", "NA")
            conference = classification.split(": ")[-1].strip().upper()
            publishable = 1 if conference != "NA" else 0
            writer.writerow([paper_id, publishable, conference, rationale])
    print(f"Results have been saved to {output_filename}")
