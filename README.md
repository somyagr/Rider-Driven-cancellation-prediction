# Rider-Driven-cancellation-prediction
Given the order and rider details, the challenge was to create a model that can predict rider-driven cancellation in advance (i.e. before getting marked as cancelled or delivered).

The data was _highly skewed_ and filled with a lot of _NaN values_ which we tried to fill by the **Progressive KNN method** we found in a kaggle discussion which worked well for it's authors([reference link](https://www.kaggle.com/c/now-you-are-playing-with-power/discussion/300903)).

We added some features because we were left with very few of them after dropping the useless ones:
* The time difference between order time and allot time
* The time difference between order time and accept time
* The day of the week
* The hour of the day
* If it's the start of the month (Boolean)

After exploring the data, we trained various models like XGBClassifier, CatboostClassifier, KNNClassifier, etc and also their ensembles but the roc_auc_score was always less than 0.65. So we tried AutoXGB on our data which boosted our score to 0.77 on the leaderboard.

We further tried smoting(very useful method for balancing skewed data) but it was overfitting the data (gave roc_auc of 0.91 in validation and 0.7 in lb)

Lastly, we tried stacking around 20 AutoXGB models but unfortunately it didn't work for us and we were left with no time to try that again.
