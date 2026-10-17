import scipy
import csv
import statsmodels
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


models = [
    'gemma4:e4b',
    'phi4-mini:3.8b',
    'qwen3.5:4b',
    'llama3.2:3b',
    'granite4.1:3b'
]

metrics = [
    'joy',
    'sadness',
    'fear',
    'disgust',
    'anger',
    'sentiment'
]

t_test_results = "t_test_results"
alpha = 0.05
results = []

for model in models:
    file_name = model.replace(":", "_")

    for metric in metrics:

        with open(f"../data/{file_name}_pair_means.csv",
                  newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        column = f"mean_{metric}_diff"

        values = [float(row[column]) for row in rows]
        mean = sum(values) / len(values)

        t_stat, p_value = ttest_1samp(values, 0)
        results.append([model, metric, mean, t_stat, p_value])

p_values = [row[4] for row in results]

reject, bh_p_values, _, _ = multipletests(
    p_values,
    alpha=0.05,
    method="fdr_bh"
)

with open("../data/t_test_results.csv", "w", newline="") as out:
    writer = csv.writer(out)

    writer.writerow([
        "model",
        "metric",
        "significance_level",
        "mean",
        "t_stat",
        "p_value",
        "bh_p_value",
        "significant_bh"
    ])

    for i, row in enumerate(results):
        model, metric, mean, t_stat, p_value = row

        writer.writerow([
            model,
            metric,
            alpha,
            mean,
            t_stat,
            p_value,
            bh_p_values[i],
            reject[i]
        ])

print("finished!")
