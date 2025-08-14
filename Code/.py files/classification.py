import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

features_day1 = pd.read_csv("features_day1.csv")
features_day2 = pd.read_csv("features_day2.csv")
features_day3 = pd.read_csv("features_day3.csv")
features_day4 = pd.read_csv("features_day4.csv")
features_day5 = pd.read_csv("features_day5.csv")

feature_tables = [features_day1, features_day2, features_day3, features_day4, features_day5]

first_day_learned = {}

for day_index, features_day in enumerate(feature_tables, start=1):
    features_day["diff_pred_vs_control"] = features_day["Prediction"] - features_day["Control"]
    features_day["learning"] = features_day["diff_pred_vs_control"] > 0.5

    animals_learned = features_day.loc[features_day["learning"], "Animal"].tolist()

    for animal in animals_learned:
        if animal not in first_day_learned:
            first_day_learned[animal] = day_index

learning_labels = {}
for animal, day in first_day_learned.items():
    if day in [1, 2]:
        learning_labels[animal] = "fast"
    elif day in [3, 4]:
        learning_labels[animal] = "normal"
    elif day == 5:
        learning_labels[animal] = "slow"

day1 = features_day1.copy()
day1["label"] = day1["Animal"].map(learning_labels)
day1 = day1.dropna(subset=["label"])  # Keep only animals with labels

X = day1.drop(columns=["Animal", "label"])
y = day1["label"]

pipeline_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=4)),
    ("clf", RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42))
])

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

scores_rf = cross_val_score(pipeline_rf, X, y, cv=cv, scoring='accuracy')

y_pred_rf = cross_val_predict(pipeline_rf, X, y, cv=cv)

cm_rf = confusion_matrix(y, y_pred_rf, labels=["fast", "normal", "slow"])
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["fast", "normal", "slow"])

clf_fi = RandomForestClassifier(random_state=42, n_estimators=200)
clf_fi.fit(X, y)

importances = clf_fi.feature_importances_
feature_names = X.columns

fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)