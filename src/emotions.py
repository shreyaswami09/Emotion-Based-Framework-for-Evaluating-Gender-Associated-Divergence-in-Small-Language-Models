from ibm_watson.natural_language_understanding_v1 \
    import Features, ConceptsOptions, SentimentOptions, EmotionOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import NaturalLanguageUnderstandingV1
import csv
import json
import sys

csv.field_size_limit(sys.maxsize)


# Authentication
apikey = "<API-KEY>"

authenticator = IAMAuthenticator(apikey)

nlu = NaturalLanguageUnderstandingV1(
    version="2019-07-12",
    authenticator=authenticator
)

nlu.set_service_url(
    "<SVC_URL>"
)

models = [
    'gemma4:e4b',
    'phi4-mini:3.8b',
    'qwen3.5:4b',
    'llama3.2:3b',
    'granite4.1:3b'
]

for model in models:

    safe_name = model.replace(":", "_")

    raw_filename = (
        f"../data/{safe_name}_results.csv"
    )

    emotion_filename = (
        f"../data/{safe_name}_emotion_results.csv"
    )

    with open(raw_filename, "r", newline="", encoding="utf-8") as raw_file:
        reader = csv.DictReader(raw_file)

        with open(emotion_filename, "w", newline="", encoding="utf-8") as emotion_file:
            writer = csv.writer(emotion_file)

            writer.writerow([
                "prompt",
                "run",
                "output",
                "joy",
                "sadness",
                "fear",
                "disgust",
                "anger",
                "sentiment_direction",
                "sentiment"
            ])

            for row in reader:
                response = nlu.analyze(
                    text=row["output"],
                    features=Features(
                        sentiment=SentimentOptions(),
                        emotion=EmotionOptions(),
                    )
                ).get_result()

                emotions = response["emotion"]["document"]["emotion"]

                sentiment_label = response["sentiment"]["document"]["label"]
                sentiment_score = response["sentiment"]["document"]["score"]

                writer.writerow([
                    row["prompt"],
                    row["run"],
                    row["output"],
                    emotions["joy"],
                    emotions["sadness"],
                    emotions["anger"],
                    emotions["fear"],
                    emotions["disgust"],
                    sentiment_label,
                    sentiment_score
                ])

    print(f"Saved IBM emotion results for {model}")
