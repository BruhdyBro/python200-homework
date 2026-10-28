import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib
from pathlib import Path

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

parent = Path(__file__).resolve().parent
outputs_folder = parent / "outputs"
models_folder = parent / "models"



# --- Preprocessing ---
#   -=- Q1 -=- 
print("-=-=- Preprocessing Section -=-=-")
scalar_q1 = StandardScaler()
scalar_q1.fit(X_train)

X_train_scaled = scalar_q1.transform(X_train)
X_test_scaled = scalar_q1.transform(X_test)

print("Mean of each columns:", X_train_scaled.mean(axis=0))
print()

# You only fit the training data so that the test data can be unseen. Fitting to the test
# data then trying to test on that data would give you increased accuracy as it's already
# seen that data before.


#   -=- Q2 -=-

encoder_q2 = OneHotEncoder(sparse_output=False)

encoded_seasons = encoder_q2.fit_transform(df[['season']])
print("Encoded Season Shape:", encoded_seasons.shape)
print("Encoder Feature Names:", encoder_q2.get_feature_names_out())
print()

# The season column becomes 4 different columns because One-Hot Encoding creates a column
# per category. Since there are 4 seasons, or 4 categories, there becomes 4 columns.


#   -=- Q3 -=-

features = numeric + ["season"]
X_full = df[features]
 
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)
 
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(), ["season"]),
    ]
)
 
pipe = Pipeline([
    ("preprocess", preprocessor),
    ("clf", LogisticRegression()),
])
 
pipe.fit(X_train_f, y_train_f)
y_pred_f = pipe.predict(X_test_f)
print("Test Accuracy:", accuracy_score(y_test_f, y_pred_f))
# The pipeline automatically fits  the scalar and encoder and transforms the training data 
# with the correct columns. It also makes sure that it is only used on training data and
# is not able to be used on test data.
print()





# --- KNN ---
#   -=- Q1 -=- 
print("-=-=- KNN Section -=-=-")

knn_pipeline = Pipeline([
    ("scalar", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
knn_pipeline.fit(X_train, y_train)
knn_y_pred = knn_pipeline.predict(X_test)
print("Scaled Accuracy:", accuracy_score(y_test, knn_y_pred))
print(classification_report(y_test, knn_y_pred))
print()


#   -=- Q2 -=- 

knn_values = [1, 3, 5, 7, 9, 11, 15, 21]

for value in knn_values:

    # build pipeline
    curr_knn_pipeline = Pipeline([
        ("scalar", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=value))
    ])


    curr_cv_score = cross_val_score(curr_knn_pipeline, X_train, y_train, cv=5)

    print("CV Score for k =", value)
    print(curr_cv_score)
    print(f"Mean: {curr_cv_score.mean():.3f}")
    print(f"Std: {curr_cv_score.std():.3f}")
    print()

# The K value i would choose is 11 as it has the highest Mean and is the point where the mean
# starts to lower back down. Choosing a small K is too sensitive to noise and overfits on the
# data, while a too large K underfits on the data.


#   -=- Q3 -=- 

# creating and fitting unscaled KNN
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
unscaled_y_pred = knn_unscaled.predict(X_test)
print("Unscaled accuracy:", accuracy_score(y_test, unscaled_y_pred))


# creating and fitting scaled KNN
scaled_knn_pipeline = Pipeline([
    ("scalar", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

scaled_knn_pipeline.fit(X_train, y_train)
scaled_y_pred = scaled_knn_pipeline.predict(X_test)
print("Scaled Accuracy:", accuracy_score(y_test, knn_y_pred))
print()

# The scaling raised the accuracy from .925 to .958 which means it helped improve the accuracy.
# Factors that are measured, such as wind speed and amount of rain, have values that differ in 
# range by a large factor. This causes the smaller range values to be represented much less and
# not properly help predict. By having the values be scaled, their distance is fairly counted.





# --- Logistic Regression ---
#   -=- Q1 -=- 
print("-=-=- Logistic Regression Section -=-=-")

logi_model = Pipeline([
    ("scalar", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])

logi_model.fit(X_train, y_train)

logi_y_pred = logi_model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, logi_y_pred):.3f}")
print(f"Precision: {precision_score(y_test, logi_y_pred, pos_label=1):.3f}")
print(f"Recall: {recall_score(y_test, logi_y_pred, pos_label=1):.3f}")
print(f"F1 Score: {f1_score(y_test, logi_y_pred, pos_label=1):.3f}")
print()


#   -=- Q2 -=- 
logreg = logi_model.named_steps["logreg"]
for name, coef in zip(features, logreg.coef_[0]):
    print(f"{name:20s}: {coef:+.2f}")
print()

# -=- Q3 -=-
probability = logi_model.predict_proba(X_test)[:, 1]

for prob in probability[:5]:
    print(f"Probability: {prob:.3f}")





# --- Evalutation ---
#   -=- Q1 -=- 
cm = confusion_matrix(y_test, logi_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["skip", "good"])
disp.plot(colorbar=False)
plt.title("Logistic Regression Confusion Matrix")
plt.savefig(outputs_folder / "logreg_confusion_matrix.png")

# There were 9 flase positives and 7 false negatives. For a Running App, a false positive
# would be worse as it predicts a good day to run but then if you go for a run, it would
# start raining on the user which is much worse than if the app said it's a bad day to run
# even though it doesn't have bad conditions. As the saying goes, "better safe than sorry".




# --- Joblib ---
#   -=- Q1 -=- 
joblib.dump(logi_model, models_folder / "warmup_model.pkl")
loaded = joblib.load(models_folder / "warmup_model.pkl")

assert (loaded.predict(X_test) == logi_model.predict(X_test)).all()
print("Predictions match.")

# saving only the Logistic Regression without the scalar would have silently wrong data.
# It would use the numbers and give the data, but they aren't scaled properly so the 
# predictions would be wrong.