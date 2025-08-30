# 🗂️ Sprint-1 Data Card — Data Understanding & Preparation

**Dataset:** judge-1377884607_tweet_product_company.csv
**Generated at:** 2025-08-30T14:26:52.808009Z

## 🎯 Sprint Goal
Prepare a clean, high-quality dataset that is reliable for baseline sentiment modeling in Sprint-2.

## 📊 Structure
- Rows (raw): 9093
- Rows (clean): 9066
- Columns (raw): 3
- Columns (clean): 5
- Duplicate tweets (raw): 27
- Duplicate tweets (clean): 0

## 🎭 Labels
- Distribution (raw): {'No emotion toward brand or product': 5389, 'Positive emotion': 2978, 'Negative emotion': 570, "I can't tell": 156}
- Distribution (clean): {'No emotion toward brand or product': 5373, 'Positive emotion': 2968, 'Negative emotion': 569, "I can't tell": 156}

## 🏷️ Targets
- Top 10 (raw): {nan: 5802, 'iPad': 946, 'Apple': 661, 'iPad or iPhone App': 470, 'Google': 430, 'iPhone': 297, 'Other Google product or service': 293, 'Android App': 81, 'Android': 78, 'Other Apple product or service': 35}
- Top 10 (clean): {nan: 5786, 'iPad': 943, 'Apple': 659, 'iPad or iPhone App': 469, 'Google': 428, 'iPhone': 296, 'Other Google product or service': 293, 'Android App': 80, 'Android': 77, 'Other Apple product or service': 35}

## 📝 Content Diagnostics (Raw)
- chars_mean: 104.95106125591114
- chars_median: 109.0
- chars_p95: 142.0
- words_mean: 17.76355438249203
- words_median: 18.0
- words_p95: 25.399999999999636

## 🧹 Cleaning Actions
- Dropped duplicate tweets
- Dropped rows with missing labels
- Normalized brand/product targets into target_norm
- Created cleaned text column text_clean

## 📌 Assumptions
- Labels are mostly accurate but may contain sarcasm/noise
- Target normalization mapping covers common ambiguous categories

## ⚠️ Risks
- Label imbalance may bias model training
- Dataset biased toward event-specific tweets (SXSW)

## 🚀 Next Sprint Gate (Sprint-2)
- baseline_model_target: ≥70% accuracy or Macro-F1
- brand_coverage_target: ≥90% normalized targets
