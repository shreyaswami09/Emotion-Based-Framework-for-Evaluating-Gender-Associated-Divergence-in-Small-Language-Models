import csv

models = [
    'gemma4:e4b',
    'phi4-mini:3.8b',
    'qwen3.5:4b',
    'llama3.2:3b',
    'granite4.1:3b'
]

for model in models:
    file_name = model.replace(":", "_")

    with open(f"../data/{file_name}_emotion_results.csv",
              newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    runs = sorted(set((row["run"]) for row in rows), key=int)

    differences_filename = f"{file_name}_differences.csv"

    with open(f"../data/{differences_filename}", "w", newline="", encoding="utf-8") as differences_file:
        writer = csv.writer(differences_file)

        writer.writerow([
            "pair_id",
            "run",
            "joy_diff",
            "abs_joy_diff (f-m)",
            "sadness_diff",
            "abs_sadness_diff (f-m)",
            "fear_diff",
            "abs_fear_diff (f-m)",
            "disgust_diff",
            "abs_disgust_diff (f-m)",
            "anger_diff",
            "abs_anger_diff (f-m)",
            "sentiment_diff",
            "abs_sentiment_diff (f-m)"
        ])

        for run in runs:
            block = [row for row in rows if row["run"] == run]

            for i in range(0, len(block), 2):

                row1 = block[i]
                row2 = block[i + 1]
                pair_index = i // 2

                if pair_index < 10:
                    male_row = row1
                    female_row = row2
                else:
                    male_row = row2
                    female_row = row1

                joy_diff = float(female_row["joy"]) - float(male_row["joy"])
                sadness_diff = float(
                    female_row["sadness"]) - float(male_row["sadness"])
                fear_diff = float(female_row["fear"]) - float(male_row["fear"])
                disgust_diff = float(
                    female_row["disgust"]) - float(male_row["disgust"])
                anger_diff = float(
                    female_row["anger"]) - float(male_row["anger"])

                sentiment_diff = float(
                    female_row["sentiment"]) - float(male_row["sentiment"])

                writer.writerow([
                    pair_index + 1,
                    run,
                    joy_diff,
                    abs(joy_diff),
                    sadness_diff,
                    abs(sadness_diff),
                    fear_diff,
                    abs(fear_diff),
                    disgust_diff,
                    abs(disgust_diff),
                    anger_diff,
                    abs(anger_diff),
                    sentiment_diff,
                    abs(sentiment_diff)
                ])

    print(f"finished {model}!")
