# Analysis

## 1. Where does the model show strong confidence (high or low predicted CTR)?

Look at your prediction histogram:

Are most predictions clustered near 0.1–0.2?
→ The model thinks the vast majority of impressions are low-probability events.

Are there pockets near 0.8–0.9?
→ The model has discovered segments that — in the limited dataset — are much more likely to click.

💡 Business interpretation:
High-confidence, high-CTR pockets represent valuable inventory segments.
Low-confidence, low-CTR pockets represent wasteful or low-value impressions.

Questions to ask yourself:

Which features (C or I columns) correspond to the “high CTR pockets”?

Do those features correspond to a publisher site? a user segment? an ad category?

If you were allocating budget: Would you spend more on those pockets?

## 2. Explore how predicted CTR varies across important features

## 3. Are predictions stable or noisy across categories?

## 4. Do certain features dominate the prediction?

## 5. Ask yourself: What would an advertiser do with these insights?

## 6. The deeper insight: CTR models are preference detectors

## Final Socratic Question (your next breakthrough)


# Implementation

## 1. Precompute "good" categorical values

### "Rare Category Handling"

- Create sorted sets for categorical columns imp<Cx>
  - include = values with >= 7 impressions
  - popular = values with score >= 60 impressions
- Use frequency thresholds to decide categorical values
  - If too rare, we will likely overfit if used directly
  - If reasonably frequent, they are safe to use
  - If very popular, we may want to handle separately

## 2. Group features into semantic "namespaces"

### Define catgorical features

```python
featureGroupA = {"C3", "C4", "C12", "C16", "C21", "C24"}
featureGroupB = {"C2", "C15", "C18"}
featureGroupC = {"C7", "C13", "C11"}
featureGroupD = {"C6", "C14", "C17", "C20", "C22", "C25", "C9", "C23"}

```
### Define numeric groups

```python

featureGroupP = {"I4", "I8", "I13"}
```

### Map into VW namespaces

We want to be able to put featuers into separate namespaces.

- Apply a different regularization per group
- Capture different roles ("primary numeric", "aux numeric", "popular catgorical in group A")

Group features by role and treat them differently.

## 3. Apply Numerical Feature Engineering

Here is the feature engineering code to transform and cap numeric features.

1. Log transform and capping
   - We appply a log transform log1p(xx) to compress heavy-tailed distributions
   - Set numCaps to the upper bound to avoid giant values that dominate the model
2. SOS2 encoding (piecewise linear interpolation)
   - Take the transformed y vcalue
   - Find two integer bins low and high around it
   - Assign weights so each feature is represented as weighted combos of the two neighboring bins

## 4. Popular vs Basic vs Missing Categorical Features

Ideas here:

- Popular categories get their own special namespace (|a, |b, |c, …). 
- Medium-frequency categories go into a generic namespace |z. 
- Extremely rare or invalid ones are not used individually; instead, they contribute to a “missingness / noise” feature |m miss:<value>.

This is sophisticated frequency-aware categorical handling:
- avoid overfitting on rare categories, 
- still capture the fact that “this row has many unknown or suspicious categories.”

Contrast with your approach:

- You used factorize per column (each unique string → integer ID) with no frequency logic. 
- They’re layering on data-dependent frequency filtering and richer categorical encoding.

# References

## Binning and Hashing

https://youtu.be/SLqKepl9rEo?si=lMSNBzW_snDdezRX

## Advertising Challenge

https://github.com/songgc/display-advertising-challenge

https://ailab.criteo.com/ressources/

https://www.kaggle.com/competitions/criteo-display-ad-challenge/overview
