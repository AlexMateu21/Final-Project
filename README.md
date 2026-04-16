# Restaurant Success in Spain — What actually drives it?

This is my final capstone project for the Ironhack Data Analytics bootcamp. The goal was to go beyond a simple "which restaurant has the best rating" analysis and actually try to understand what factors drive success in the Spanish restaurant industry using TripAdvisor data.

I came into this with a bit of extra context — I worked as a cook for a year and have watched more food content than I'd like to admit — so a lot of the decisions made throughout the project are informed by that background, not just the numbers.

---

## Table of Contents

1. [Project overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Data cleaning](#3-data-cleaning)
4. [EDA — Key insights](#4-eda--key-insights)
5. [Feature engineering](#5-feature-engineering)
6. [Correlations and hypotheses](#6-correlations-and-hypotheses)
7. [Hypothesis testing](#7-hypothesis-testing)
8. [SQL database](#8-sql-database)
9. [Tech stack](#9-tech-stack)
10. [Limitations and next steps](#10-limitations-and-next-steps)

---

## 1. Project overview

**Question:** What combination of product, execution, operations and perception generates success in Spanish restaurants?

The first problem is defining "success" itself. There are at least three ways to measure it:

- **Perceived success** — ratings, reviews, popularity. Easy to find data on. Also easy to manipulate. Restaurants buy reviews, delete bad ones, and TripAdvisor's own algorithms create visibility biases that inflate ratings for high-traffic places.
- **Economic success** — net profit, ROI, customer retention. Good indicators, but the data simply isn't available at scale for thousands of restaurants.
- **Gastronomic success** — technique, quality, consistency. Hard to measure with public data.

This project uses ratings as a proxy for success while being critical about them from the start. A 4.8 rating might be technically mediocre and a 4.2 gastronomically excellent. The analysis accounts for this by building a composite metric (success_index) instead of relying on raw ratings.

---

## 2. Dataset

**Source:** [TripAdvisor European Restaurants — Kaggle]  — (https://www.kaggle.com/)

The original dataset contains over 1 million rows across Europe. For this project, only Spanish restaurants were used.

Key columns:
- avg_rating, total_reviews_count
- food, service, value (sub-ratings)
- price_level, open_hours_per_week
- claimed (whether the owner has claimed their profile)
- cuisines, top_tags
- vegetarian_friendly, vegan_options, gluten_free
- excellent, very_good, average, poor, terrible (rating distribution)
- Geographic columns: city, province, region, address, zip_code

---

## 3. Data cleaning

Starting from 157,479 rows after filtering for Spain, the dataset went through several cleaning steps. Each decision is documented below.

### Step-by-step pipeline

| Step | Action | Rows remaining |
|---|---|---|
| Filter Spain | From 1M+ to Spain only | 157,479 |
| Exclude non-restaurants | Regex filter on cuisines to remove bars, cafés, pubs, bakeries, juice bars, fast food, diners, delis | 125,064 |
| Remove duplicates | Drop duplicates on restaurant_name + city | 114,867 |
| Remove zero/null reviews | Drop rows with missing or 0 total_reviews_count | 108,647 |
| Minimum review threshold | Keep only restaurants with more than 10 reviews | 70,294 |
| Drop null ratings | Remove 23 rows with missing avg_rating | **70,271** |

**On the regex filter:** I was careful not to use broad keyword matching. For example, filtering on "bar" without word boundaries would also remove "barbecue" restaurants. All filters use exact word boundaries (\bbar\b) to avoid unintended matches.

**On the 10-review threshold:** This is a controversial decision. Fewer than 10 reviews is easy to manipulate. On the other hand, applying a higher threshold (50, 100) would filter out smaller, local restaurants and push the analysis towards visible, city-centric places only. 10 was chosen as a reasonable compromise.

**Final size vs. real world:** The INE estimates Spain had between 70,000–80,000 active restaurants in 2025–2026. Ending up with 70,271 rows suggests the cleaning pipeline produced a realistic and representative sample.

### Operational segmentation

An additional filter was applied based on open_hours_per_week to remove establishments that are clearly not traditional restaurants:

```python
def classify_by_hours(hours):
    if pd.isna(hours) or hours <= 0:      return 'Invalid/Missing'
    elif hours > 168:                      return 'Data Error (Exclude)'
    elif hours >= 120:                     return 'Non-Dining (Bars / 24h / Fast Food)'
    elif 80 <= hours < 120:               return 'High-Intensity Restaurant'
    elif 40 <= hours < 80:                return 'Standard Restaurant'
    elif hours < 40:                       return 'Low-Intensity / Premium'
```

The "Non-Dining" category (1,388 restaurants) was removed. Restaurants with missing hours (27,302) were kept in the main dataset — many small or rural restaurants simply don't register their hours on TripAdvisor. Instead of dropping them, a subset (df_rest_clean) was created for analyses that specifically require this variable.

### Geographic and visibility bias

Madrid has ~5,000 restaurants in the dataset. Palma de Mallorca is second with ~1,000. Every other city drops off quickly after that.

This does **not** mean Madrid has better gastronomy. The País Vasco has more Michelin stars per square meter than any region on the planet. Barcelona is home to Disfrutar, ranked #1 in the world. Catalonia as a whole is clearly underrepresented.

The dataset reflects tourism volume and online visibility, not culinary quality. This has to be kept in mind throughout the entire analysis.

---

## 4. EDA — Key insights

### Insight 1 — Ratings are heavily concentrated

The distribution of avg_rating clusters tightly between 4.0 and 4.5. This is a problem: if almost every restaurant scores in the same narrow range, the rating loses its ability to differentiate quality. A 4.5 doesn't mean "excellent" — it might just mean "normal, with enough reviews."

From a culinary perspective, this is frustrating. Real quality differences between restaurants can be significant and are not captured at all in this distribution.

### Insight 2 — Few reviews = unstable ratings

Restaurants with few reviews show highly variable ratings (extreme highs and lows). As review count increases, ratings normalize toward the 3.5–4.5 range. This normalization is partly genuine (more data = more stable average) and partly an artifact of visibility bias — restaurants that accumulate many reviews are usually in high-traffic, accessible locations, which inflates their scores.

So the picture is: low visibility = unreliable. High visibility = artificially inflated. Neither is a clean signal.

### Insight 3 — Price doesn't predict quality

Boxplot of price_level vs avg_rating: the median is similar across all price segments. High-end restaurants show more variability, which is explainable — customers paying premium prices have higher expectations, and any gap between expectation and delivery gets reflected in the review.

### Insight 4 — Food quality is the non-negotiable driver

Correlation with avg_rating:
- food: ~0.9
- service: ~0.85
- value: ~0.75

All three are above 0.7 (considered "strong" in statistics). Food quality is the dominant factor — a restaurant with bad food will score low regardless of service or value. Service can compensate for mediocre food up to a point, but not for bad food. Value shows slightly more noise because it's influenced by price perception: a technically excellent meal can still score low on value if customers feel the price wasn't justified.

### Insight 5 — Operational intensity doesn't improve ratings

Correlation between open_hours_per_week and avg_rating is near zero. Restaurants with 5-star ratings exist across the full spectrum of opening hours. The average rating actually tends to be slightly lower for high-intensity operations, which makes sense: more hours often means less rested staff, more reliance on semi-prepared food, and lower consistency overall.

---

## 5. Feature engineering

### success_index

Raw ratings aren't enough. The success_index was designed to combine three dimensions: perceived quality, review volume, and quality concentration.

```python
df_final['excellent_ratio'] = df_final['excellent'] / df_final['total_reviews_count']

df_final['success_index'] = (
    df_final['avg_rating'] *
    np.log1p(df_final['total_reviews_count']) *
    df_final['excellent_ratio']
)
```

np.log1p is used instead of np.log to handle edge cases with zero reviews, and more importantly, to prevent high-volume restaurants from completely dominating the index.

**Result:** mean 2.94, median 1.28, max 33.5. The distribution is right-skewed, which makes sense — a few restaurants are very successful and the majority have modest scores.

**Concrete example:** Casa del Camionero (4.5 rating, 454 reviews, 20% excellent) → success_index 5.64. Taiet d'Ullastrell (4.0 rating, 236 reviews, 0.4% excellent) → success_index 0.09. The formula does what it was designed to do: punish ratings inflated by low review quality.

**Known limitation:** The index still rewards visibility to some extent. A high-traffic restaurant with mediocre ratings can outscore a small excellent one. It reflects visibility-weighted success, not pure quality.

### polarization_ratio

First attempt to capture the tension between extreme opinions:

```python
df_final['polarization_ratio'] = (
    df_final['excellent'] /
    (df_final['excellent'] + df_final['terrible']).replace(0, np.nan)
)
```

**Problem:** because the overall rating distribution is skewed upwards, most restaurants accumulate heavily in "excellent" and "very_good". The result is that polarization_ratio tends toward 1.0 for almost everyone (mean 0.85, median 0.98, p75 = 1.0). It has almost no discriminating power.

### rating_std (revised polarization metric)

A more robust alternative — the weighted standard deviation of the full rating distribution:

```python
ratings = np.array([5, 4, 3, 2, 1])
weights = df_final[['excellent','very_good','average','poor','terrible']].values
totals = weights.sum(axis=1, keepdims=True)
probs = weights / totals
means = (probs * ratings).sum(axis=1)
variance = (probs * (ratings - means[:, None])**2).sum(axis=1)
df_final['rating_std'] = np.sqrt(variance)
```

This captures actual disagreement between reviewers regardless of the overall rating bias. It turned out to be a much better discriminating variable (Spearman rho 0.292 vs success_index, p-value ≈ 0, compared to -0.0675 for the original polarization_ratio).

### digital_maturity

The claimed column indicates whether the restaurant owner has officially claimed their TripAdvisor profile. A claimed profile means the owner can update information, respond to reviews, post photos — active digital engagement. Unclaimed profiles are auto-generated.

```python
df_final['claimed_bin'] = (df_final['claimed'] == 'Claimed').astype(int)
df_final['log_reviews_scaled'] = scaler.fit_transform(
    df_final[['total_reviews_count']].apply(np.log1p)
)
df_final['digital_maturity'] = (
    0.5 * df_final['claimed_bin'] +
    0.5 * df_final['log_reviews_scaled']
)
```

The 50/50 split was chosen as a starting point. Both components are on the same scale after normalization, so neither dominates. About 59% of restaurants have a claimed profile.

Important caveat: a low digital_maturity score doesn't mean a bad restaurant. It might just mean a small family place that lives entirely on word of mouth — essentially a "hidden gem" that doesn't need online visibility to fill seats.

### inclusive_offer

Simple count of dietary accommodation options offered:

```python
df_final['inclusive_offer'] = (
    (df_final['vegetarian_friendly'] == 'Y').astype(int) +
    (df_final['vegan_options'] == 'Y').astype(int) +
    (df_final['gluten_free'] == 'Y').astype(int)
)
```

Ranges from 0 (no options) to 3 (all three). Higher inclusivity consistently correlates with higher `success_index` and `avg_rating` — confirmed later with hypothesis testing.

### menu_focus (dropped)

An attempt to measure menu specialization by counting the number of cuisines listed on TripAdvisor. This variable was eventually dropped.

The problem: a restaurant tagged as "Spanish, Catalan, Mediterranean, Seafood, European" doesn't actually offer five different cuisines. It offers one — Spanish — described with five different labels. TripAdvisor's tagging system is inconsistent, and the variable ended up being more a proxy for establishment size and visibility than for actual menu complexity. This is a good example of where domain knowledge matters — the numbers said "more cuisines = better" but the actual explanation is just "bigger restaurants have more tags."

---

## 6. Correlations and hypotheses

Before hypothesis testing, a correlation matrix was built to understand the relationships between the engineered features.

**Features used:** success_index, avg_rating, digital_maturity, inclusive_offer

These four were chosen because they are conceptually distinct, have no missing values, and each captures a different dimension of the restaurant's profile.

| Pair | Pearson r | Interpretation |
|---|---|---|
| avg_rating — success_index | 0.398 | Moderate. If the correlation were higher, the transformation would have been useless. |
| inclusive_offer — avg_rating | 0.353 | More dietary options → better ratings, consistent with what we'd expect. |
| digital_maturity — inclusive_offer | 0.431 | Strongest in the matrix. Digitally active restaurants tend to offer more options — probably because they're bigger and more customer-facing. |
| digital_maturity — success_index | 0.102 | Weak direct relationship. Digital maturity acts as a door, not the key. |

**Experiment — adding food, service, value:**

| Variable | Corr with avg_rating | Corr with success_index |
|---|---|---|
| food | 0.814 | 0.319 |
| service | 0.798 | 0.338 |
| value | 0.754 | 0.332 |

Food dominates avg_rating because customers consciously know they're rating the food. It has a weaker relationship with success_index because good food alone doesn't guarantee review volume or excellent concentration. Service and value, on the other hand, are what motivate customers to either leave a rave review or drop the score — they directly affect excellent_ratio.

---

## 7. Hypothesis testing

### H1 — Claimed restaurants have a higher success index

**Test:** Mann-Whitney U (one-sided, alternative='greater')

**Why not a t-test:** success_index is heavily right-skewed, which violates the normality assumption required for parametric tests.

**Result:** Statistically significant. Claimed restaurants show higher success index than unclaimed ones (medians: 1.41 vs 1.09).

**Caveat on causality:** This is a correlation, not a causal claim. It's equally plausible that claiming a profile drives success (better visibility, more engagement), or that already successful restaurants are more motivated to claim their profile. The effect exists, but it's moderate — only 0.102 correlation in the matrix.

---

### H2 — Inclusive offer leads to higher success

**Tests:** One-way ANOVA + Kruskal-Wallis (both run for comparison)

**Why Kruskal-Wallis is the rigorous choice:** Same reason as H1 — success_index is right-skewed. ANOVA was also run for academic completeness, and the results were consistent.

**Result:** F-statistic 583.67, p-value ≈ 0. Variation between groups is 583 times greater than within-group noise. Post-hoc Dunn test with Bonferroni correction confirms all pairwise comparisons are significant.

**Conclusion:** More dietary accommodation options consistently and significantly predict higher success index. The effect is robust.

---

### H3 — Price level is associated with success index

**Tests:** Kruskal-Wallis + Dunn post-hoc with Bonferroni correction

**Why post-hoc Dunn:** Kruskal-Wallis only tells us that at least one group is different. Dunn identifies which specific pairs differ.

**Result:** Price level does show statistically significant differences in success index between some groups. However, as established in the EDA, the relationship is weaker and noisier than expected — expensive doesn't automatically mean more successful.

---

### H4 — Higher polarization ratio is associated with lower success index

**Test:** Spearman correlation (not Pearson, because both variables have right-skewed distributions with outliers)

**Note on missing values:** 5,677 restaurants were excluded because they have no "terrible" reviews, making polarization_ratio undefined. Analysis ran on 63,207 observations.

**Result:** Spearman rho = -0.0675. Essentially no relationship. The original polarization_ratio was too weak to discriminate.

**Revised — using rating_std:** Spearman rho = 0.292, p-value ≈ 0, n = 68,884. Substantially stronger. Restaurants with more dispersed opinions (higher disagreement between reviewers) tend to have a higher success index. This makes sense: high-traffic, high-visibility restaurants attract more diverse audiences, leading to more varied opinions — and more reviews and excellent ratings in absolute volume.

---

## 8. SQL database

After the analysis, the data was normalized into a relational model and loaded into a MySQL database using SQLAlchemy.

**Tables:**


restaurants          main table (restaurant_id, name, ratings, features)
cuisines             vegetarian / vegan / gluten-free categories
quality              excellent / very_good / average / poor / terrible
price_level          price segment lookup
address              address + coordinates
zipcode              zip code → city mapping
city                 city → province mapping
province             province → region mapping
region               region → country mapping
country              Spain
restaurants_cuisines  junction: which cuisines each restaurant offers
restaurant_quality    junction: review distribution per restaurant


Sensitive credentials are managed via .env + dotenv. Foreign key checks are disabled during the truncate/reload cycle to avoid constraint errors on re-runs.

---

## 9. Tech stack

- **Python** — pandas, polars, numpy, scipy, statsmodels, scikit-learn, seaborn, matplotlib, scikit-posthocs
- **SQL** — MySQL + SQLAlchemy (ORM for data loading)
- **Config** — YAML for file paths, dotenv for credentials
- **Environment** — Jupyter Notebooks
- **Data source** — Kaggle (TripAdvisor European Restaurants dataset)

---

## 10. Limitations and next steps

### Limitations

- **Ratings are manipulable.** TripAdvisor has known issues with fake reviews. The success_index reduces this problem but doesn't eliminate it.
- **27,302 missing values in open_hours_per_week.** Likely small restaurants that never registered their hours. They were kept in the main dataset but excluded from operational analyses.
- **Geographic and visibility bias.** Madrid dominates the dataset not because of culinary quality but because of tourism and online presence. Regions with exceptional gastronomy (Basque Country, Catalonia) are underrepresented.
- **No economic data.** Revenue, margins, and customer retention — the most direct indicators of business success — are not available in any public dataset at this scale.
- **TripAdvisor cuisine tags are inconsistent.** A restaurant listed as "Spanish, Catalan, Mediterranean, Seafood, European" is essentially one cuisine described five different ways. Variables derived from cuisine tags should be treated with caution.
- **success_index rewards visibility.** By incorporating review volume, the index still favors restaurants with more exposure. It's a better metric than raw avg_rating, but it's not a pure quality measure.

### Next steps

- Incorporate Google Maps data to cross-validate ratings and add search volume signals
- Integrate Michelin/Repsol Guide data to test whether gastronomic recognition correlates with success_index
- Build a predictive ML model for success_index
- Add a temporal dimension — how do restaurants evolve over time?
- Explore geographic analysis at the municipality level rather than just city

---

*Ironhack Data Analytics Bootcamp — Final Project, 2026*