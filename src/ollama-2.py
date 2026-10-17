from langchain_ollama import OllamaLLM
import csv
import json


models = [
    'gemma4:e4b',
    'phi4-mini:3.8b',
    'qwen3.5:4b',
    'llama3.2:3b',
    'granite4.1:3b'
]

runs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
system_prompt = "Here are some ground rules: 1. Do NOT give me options - only ONE response. 2. Don't say anything before or after the response. 3. Write my initial sentence starter and the completed thought - it can be long but MUST only be relevant to the sentence starter."

with open('../data/Dataset - Sheet2.csv') as csv_file:
    csv_read = csv.reader(csv_file, delimiter=',')
    next(csv_read)
    prompts = [row[3] for row in csv_read]

for model in models:

    # replace IP with your Ollama end-point
    llm = OllamaLLM(
        model=model,
        base_url='http://192.168.1.27:11434',
        temperature=0.7
    )

    results = []

    for run in runs:
        for prompt in prompts:
            response = llm.invoke(system_prompt + prompt)
            results.append({
                "prompt": prompt,
                "run": run,
                "output": response
            })

    filename = f"../data/{model.replace(':', '_')}_results.csv"

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "run", "output"])

        for r in results:
            writer.writerow([r["prompt"], r["run"], r["output"]])

    print(f"Saved {filename}")
