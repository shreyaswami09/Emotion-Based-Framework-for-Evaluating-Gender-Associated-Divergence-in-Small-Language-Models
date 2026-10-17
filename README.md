## Ethics of AI: A Framework for Evaluating Gender-Associated Emotional Divergence in Small Language Models

This paper presents a methodology to statistically evaluate and quantify gender-associated emotional divergence in small language models (SLMs).

It does so through replication, t-testing, and multiple test correction to determine significance. Metrics are also calculated using magnitude of the emotional divergence.

This framework compliments existing frameworks, like the Vacuous Neutrality Framework, with a new, emotion-focused perspective and concentrates specifically on SLM behavior.

## Objective
Recently, there has been a surge in usage of language models due to the plummeting token costs. But because of their cost-efficiency and localization, much of the industrial usage has shifted from large to small language models. However, the this shift has not been paralleled by equivalent evaluation in SLMs' multi-faceted ethical concerns. As a result, much of the current testing performed on SLM objectivity is carried out using LLM frameworks. 

This paper presents a statistical method of benchmarking to evaluate consistent divergence in SLM responses based on gender-association. Furthermore, this framework relies solely on emotional and sentiment connotations of responses, a relatively-less thoroughly researched aspect of existing SLM benchmarks. It consists of a procured prompt dataset consisting of paired prompts, differing only in gender-associated name. Through replication, t-testing, and multiple t-test correction, results reveal that the five SLMs selected for this study do not exhibit significant consistencies towards any gender based on emotional pull. 

Furthermore, the magnitudes of emotional differences between pairs were consolidated to produce emotion and sentiment metrics for the five SLMs. Since the metric values were not negligibly low, they indicate that even if there is no consistent tendency towards one gender, stochastic volatility may still contribute to a model's response being skewed. Although those might be outliers, they still hold potential risks for the integrity of companies employing them to one-time users who happen to receive a said outlier. 

Logical next steps of this experiment are to perform it with a much larger sample size, with multiple emotion evaluators, and should be replicated for LLMs to compare outputs and potentially reveal patterns that suggest SLMs should be evaluated differently. 
