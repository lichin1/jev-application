# Kaggle × Jev benchmark results

Every method is scored on the whole 20% test split (`Full test n` rows), never seen in training.
Kaggle is the classic Kaggle solution trained on the 80% train split. Jev 0-shot never sees training labels;
Jev k-shot adds k random labeled training examples per class to each answer's criteria; Jev cluster-shot takes
one example per (label, k-means cluster) cell; Jev matched random is its control with the same per-label counts.
Kaggle + Jev columns stack the Kaggle model's out-of-fold score with Jev's probability in a logistic regression;
Stage 2 · Kaggle only is the same logistic regression without Jev, the control to compare them with.
Examples always come from the train split. Bold and underline mark the best score in each row.

## 1. Summary

| Dataset | Full test n | Train | Problem | Jev question | State (first row) | True label |
| --- | ---: | ---: | --- | --- | --- | --- |
| [titanic](https://www.kaggle.com/competitions/titanic) | 179 | 712 | Predict whether a Titanic passenger survived | **Noul**: `passenger` describes a person aboard the RMS Titanic when it sank on 15 April 1912. Based on this profile and what is known about who reached the lifeboats, did this passenger survive? (yes / no) | `{"passenger": {"ticket_class": "3rd (lower)", "sex": "male", "title": "Mr", "age_years": 24.0, "siblings_or_sp …` | 0 |
| [sms_spam](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) | 1115 | 4457 | Detect whether an SMS message is spam | **Noul**: Is `sms` a spam text message? (yes / no) | `{"sms": "No need to buy lunch for me.. I eat maggi mee.."}` | 0 |
| [imdb](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | 10000 | 40000 | Classify a movie review as positive or negative | **Noul**: Is `review` a positive review overall, meaning the reviewer recommends the movie? (yes / no) | `{"review": "Yes, MTV there really is a way to market Daria. What started as a clever teenage angst-\"comment o …` | 0 |
| [bbc_news](https://www.kaggle.com/competitions/learn-ai-bbc) | 445 | 1780 | Assign a BBC News article to one of 5 sections | **Choice**: Which BBC News section does `article` belong to? (business / entertainment / politics / sport / tech) | `{"article": "profits jump at china s top bank industrial and commercial bank (icbc) china s biggest lender h …` | business |
| [iris](https://www.kaggle.com/datasets/uciml/iris) | 30 | 120 | Identify the iris species from 4 flower measurements | **Choice**: `flower_measurements_cm` are measurements of one iris flower. Which species is it? (setosa / versicolor / virginica) | `{"flower_measurements_cm": {"sepal_length": 4.4, "sepal_width": 3.0, "petal_length": 1.3, "petal_width": 0.2}}` | setosa |

## 2. Accuracy

| Dataset | Full test n | Kaggle | Jev 0-shot | Jev 3-shot | Jev cluster-shot | Jev matched random | Stage 2 · Kaggle only | Kaggle + Jev 0-shot | Kaggle + Jev cluster |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| titanic | 179 | **0̲.̲8̲2̲1̲** | 0.648 | 0.682 | 0.754 | 0.682 | 0.810 | 0.793 | 0.799 |
| sms_spam | 1115 | 0.985 | 0.979 | 0.980 | 0.980 | 0.981 | 0.985 | **0̲.̲9̲9̲2̲** | 0.991 |
| imdb | 10000 | 0.918 | 0.963 | 0.963 | 0.963 | 0.963 | 0.918 | **0̲.̲9̲6̲4̲** | **0̲.̲9̲6̲4̲** |
| bbc_news | 445 | 0.987 | 0.982 | 0.978 | 0.982 | 0.975 | 0.987 | 0.989 | **0̲.̲9̲9̲1̲** |
| iris | 30 | 0.933 | 0.600 | 0.900 | 0.967 | 0.833 | 0.967 | 0.967 | **1̲.̲0̲0̲0̲** |

## 3. Macro-F1

| Dataset | Full test n | Kaggle | Jev 0-shot | Jev 3-shot | Jev cluster-shot | Jev matched random | Stage 2 · Kaggle only | Kaggle + Jev 0-shot | Kaggle + Jev cluster |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| titanic | 179 | **0̲.̲8̲1̲0̲** | 0.593 | 0.644 | 0.734 | 0.644 | 0.795 | 0.777 | 0.784 |
| sms_spam | 1115 | 0.965 | 0.956 | 0.958 | 0.958 | 0.960 | 0.965 | **0̲.̲9̲8̲2̲** | 0.980 |
| imdb | 10000 | 0.918 | 0.963 | 0.963 | 0.962 | 0.963 | 0.918 | **0̲.̲9̲6̲4̲** | **0̲.̲9̲6̲4̲** |
| bbc_news | 445 | 0.986 | 0.982 | 0.978 | 0.982 | 0.975 | 0.986 | 0.989 | **0̲.̲9̲9̲1̲** |
| iris | 30 | 0.933 | 0.494 | 0.898 | 0.967 | 0.828 | 0.967 | 0.967 | **1̲.̲0̲0̲0̲** |

## 4. ROC AUC (binary tasks)

Uses the Kaggle model's positive-class score and Jev's `noul` probability.

| Dataset | Full test n | Kaggle | Jev 0-shot | Jev 3-shot | Jev cluster-shot | Jev matched random | Stage 2 · Kaggle only | Kaggle + Jev 0-shot | Kaggle + Jev cluster |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| titanic | 179 | 0.841 | 0.749 | 0.783 | 0.826 | 0.783 | 0.841 | 0.847 | **0̲.̲8̲5̲9̲** |
| sms_spam | 1115 | 0.992 | 0.991 | 0.992 | 0.989 | 0.989 | 0.992 | **0̲.̲9̲9̲4̲** | 0.993 |
| imdb | 10000 | 0.975 | 0.993 | 0.993 | 0.993 | 0.993 | 0.975 | **0̲.̲9̲9̲4̲** | **0̲.̲9̲9̲4̲** |

## Run details

- **titanic** Jev 0-shot: `{"model": "jev-latest", "shots": "0", "errors": 0, "latency_p50_s": 0.287, "latency_p95_s": 0.364, "input_tokens_total": 80056}`
- **titanic** Jev 3-shot: `{"model": "jev-latest", "shots": "3", "errors": 0, "examples_per_label": {"0": 3, "1": 3}, "latency_p50_s": 0.273, "latency_p95_s": 0.512, "input_tokens_total": 201955}`
- **titanic** Jev cluster-shot: `{"model": "jev-latest", "shots": "cluster", "errors": 0, "examples_per_label": {"0": 3, "1": 3}, "latency_p50_s": 0.28, "latency_p95_s": 0.437, "input_tokens_total": 202492}`
- **titanic** Jev matched random: `{"model": "jev-latest", "shots": "cluster-random", "errors": 0, "examples_per_label": {"0": 3, "1": 3}, "latency_p50_s": 0.273, "latency_p95_s": 0.512, "input_tokens_total": 201955}`
- **titanic** Kaggle + Jev 0-shot: `{"stage2_train_rows": 712, "jev_missing": 0, "stage2_weight": {"kaggle": 1.946, "jev": 0.247}}`
- **titanic** Kaggle + Jev cluster: `{"stage2_train_rows": 706, "jev_missing": 0, "stage2_weight": {"kaggle": 1.674, "jev": 0.56}}`
- **sms_spam** Jev 0-shot: `{"model": "jev-latest", "shots": "0", "errors": 0, "latency_p50_s": 0.291, "latency_p95_s": 0.622, "input_tokens_total": 417218}`
- **sms_spam** Jev 3-shot: `{"model": "jev-latest", "shots": "3", "errors": 0, "examples_per_label": {"0": 3, "1": 3}, "latency_p50_s": 0.286, "latency_p95_s": 0.508, "input_tokens_total": 833113}`
- **sms_spam** Jev cluster-shot: `{"model": "jev-latest", "shots": "cluster", "errors": 0, "examples_per_label": {"0": 8, "1": 5}, "latency_p50_s": 0.292, "latency_p95_s": 0.541, "input_tokens_total": 988098}`
- **sms_spam** Jev matched random: `{"model": "jev-latest", "shots": "cluster-random", "errors": 0, "examples_per_label": {"0": 8, "1": 5}, "latency_p50_s": 0.29, "latency_p95_s": 0.468, "input_tokens_total": 1117438}`
- **sms_spam** Kaggle + Jev 0-shot: `{"stage2_train_rows": 2000, "jev_missing": 0, "stage2_weight": {"kaggle": 1.892, "jev": 3.073}}`
- **sms_spam** Kaggle + Jev cluster: `{"stage2_train_rows": 2000, "jev_missing": 0, "stage2_weight": {"kaggle": 2.07, "jev": 3.189}}`
- **imdb** Jev 0-shot: `{"model": "jev-latest", "shots": "0", "errors": 0, "latency_p50_s": 0.294, "latency_p95_s": 0.445, "input_tokens_total": 6230983}`
- **imdb** Jev 3-shot: `{"model": "jev-latest", "shots": "3", "errors": 0, "examples_per_label": {"0": 3, "1": 3}, "latency_p50_s": 0.294, "latency_p95_s": 0.459, "input_tokens_total": 14350983}`
- **imdb** Jev cluster-shot: `{"model": "jev-latest", "shots": "cluster", "errors": 0, "examples_per_label": {"0": 2, "1": 2}, "latency_p50_s": 0.292, "latency_p95_s": 0.444, "input_tokens_total": 12340983}`
- **imdb** Jev matched random: `{"model": "jev-latest", "shots": "cluster-random", "errors": 0, "examples_per_label": {"0": 2, "1": 2}, "latency_p50_s": 0.3, "latency_p95_s": 0.468, "input_tokens_total": 11650983}`
- **imdb** Kaggle + Jev 0-shot: `{"stage2_train_rows": 2000, "jev_missing": 0, "stage2_weight": {"kaggle": 1.172, "jev": 4.198}}`
- **imdb** Kaggle + Jev cluster: `{"stage2_train_rows": 2000, "jev_missing": 0, "stage2_weight": {"kaggle": 0.942, "jev": 4.108}}`
- **bbc_news** Jev 0-shot: `{"model": "jev-latest", "shots": "0", "errors": 0, "latency_p50_s": 0.293, "latency_p95_s": 0.567, "input_tokens_total": 407340, "accuracy_when_confidence_ge_0.8": 0.992822966507177, "share_confidence_ge_0.8": 0.9393258426966292}`
- **bbc_news** Jev 3-shot: `{"model": "jev-latest", "shots": "3", "errors": 0, "examples_per_label": {"business": 3, "entertainment": 3, "politics": 3, "sport": 3, "tech": 3}, "latency_p50_s": 0.304, "latency_p95_s": 0.615, "input_tokens_total": 1412595, "accuracy_when_confidence_ge_0.8": 0.9928057553956835, "share_confidence_ge_0.8": 0.9370786516853933}`
- **bbc_news** Jev cluster-shot: `{"model": "jev-latest", "shots": "cluster", "errors": 0, "examples_per_label": {"business": 3, "entertainment": 4, "politics": 2, "sport": 3, "tech": 3}, "latency_p50_s": 0.302, "latency_p95_s": 0.586, "input_tokens_total": 1372990, "accuracy_when_confidence_ge_0.8": 0.9928400954653938, "share_confidence_ge_0.8": 0.9415730337078652}`
- **bbc_news** Jev matched random: `{"model": "jev-latest", "shots": "cluster-random", "errors": 0, "examples_per_label": {"business": 3, "entertainment": 4, "politics": 2, "sport": 3, "tech": 3}, "latency_p50_s": 0.298, "latency_p95_s": 0.627, "input_tokens_total": 1416155, "accuracy_when_confidence_ge_0.8": 0.9928057553956835, "share_confidence_ge_0.8": 0.9370786516853933}`
- **bbc_news** Kaggle + Jev 0-shot: `{"stage2_train_rows": 1780, "jev_missing": 0}`
- **bbc_news** Kaggle + Jev cluster: `{"stage2_train_rows": 1765, "jev_missing": 0}`
- **iris** Jev 0-shot: `{"model": "jev-latest", "shots": "0", "errors": 0, "latency_p50_s": 0.349, "latency_p95_s": 0.734, "input_tokens_total": 11910, "accuracy_when_confidence_ge_0.8": 1.0, "share_confidence_ge_0.8": 0.3333333333333333}`
- **iris** Jev 3-shot: `{"model": "jev-latest", "shots": "3", "errors": 0, "examples_per_label": {"setosa": 3, "versicolor": 3, "virginica": 3}, "latency_p50_s": 0.281, "latency_p95_s": 0.581, "input_tokens_total": 27300, "accuracy_when_confidence_ge_0.8": 0.9565217391304348, "share_confidence_ge_0.8": 0.7666666666666667}`
- **iris** Jev cluster-shot: `{"model": "jev-latest", "shots": "cluster", "errors": 0, "examples_per_label": {"setosa": 1, "versicolor": 2, "virginica": 2}, "latency_p50_s": 0.286, "latency_p95_s": 0.57, "input_tokens_total": 20940, "accuracy_when_confidence_ge_0.8": 1.0, "share_confidence_ge_0.8": 0.8}`
- **iris** Jev matched random: `{"model": "jev-latest", "shots": "cluster-random", "errors": 0, "examples_per_label": {"setosa": 1, "versicolor": 2, "virginica": 2}, "latency_p50_s": 0.277, "latency_p95_s": 0.591, "input_tokens_total": 20940, "accuracy_when_confidence_ge_0.8": 0.9565217391304348, "share_confidence_ge_0.8": 0.7666666666666667}`
- **iris** Kaggle + Jev 0-shot: `{"stage2_train_rows": 120, "jev_missing": 0}`
- **iris** Kaggle + Jev cluster: `{"stage2_train_rows": 115, "jev_missing": 0}`
