
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score


def load_and_understand_data():
   
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = [
        "Class 1",
        "Class 2",
        "Class 3"
    ]

    print("=" * 60)
    print("STEP 1: LOAD & UNDERSTAND DATASET")
    print("=" * 60)
    print(f"Total samples : {X.shape[0]}")
    print(f"Features      : {X.shape[1]} -> {feature_names}")
    print(f"Classes       : {len(class_names)} -> {list(class_names)}")
    print(f"Samples/class : {np.bincount(y)}  (balanced dataset)")
    print("\nFirst 5 rows of raw data:")
    for i in range(5):
        print(f"  {X[i]}  -> {class_names[y[i]]}")
    print()

    return X, y, feature_names, class_names


def scale_features(X_train, X_test):

    print("=" * 60)
    print("STEP 2: FEATURE SCALING (StandardScaler)")
    print("=" * 60)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train only
    X_test_scaled = scaler.transform(X_test)          # transform test using train's scale

    print(f"Before scaling -> mean: {X_train[:, 0].mean():.2f}, std: {X_train[:, 0].std():.2f}")
    print(f"After scaling  -> mean: {X_train_scaled[:, 0].mean():.2f}, std: {X_train_scaled[:, 0].std():.2f}")
    print()
    return X_train_scaled, X_test_scaled, scaler


def split_data(X, y, test_size=0.2, random_state=42):

    print("=" * 60)
    print("STEP 3: TRAIN-TEST SPLIT")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,       # remove order bias
        stratify=y           # keep class proportions equal in both sets
    )
    print(f"Training set : {X_train.shape[0]} samples")
    print(f"Testing set  : {X_test.shape[0]} samples")
    print()
    return X_train, X_test, y_train, y_test


def find_best_k(X_train, y_train, X_test, y_test, max_k=20):
  
    print("=" * 60)
    print("STEP 4: TUNING K (finding the elbow)")
    print("=" * 60)
    best_k, best_acc = 1, 0.0
    for k in range(1, max_k + 1):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        if acc > best_acc:
            best_k, best_acc = k, acc
    print(f"Best K found : {best_k}  (accuracy: {best_acc:.2%})")
    print()
    return best_k


def train_and_predict(X_train, y_train, X_test, k):

    print("=" * 60)
    print("STEP 5: TRAIN MODEL (K-Nearest Neighbors)")
    print("=" * 60)
    model = KNeighborsClassifier(n_neighbors=k)   # INSTANTIATE
    model.fit(X_train, y_train)                   # FIT (memorize the map)
    predictions = model.predict(X_test)            # PREDICT (apply logic)
    print(f"Model trained with n_neighbors={k}")
    print()
    return model, predictions


def evaluate_model(y_test, predictions, class_names):
    print("=" * 60)
    print("STEP 6: OUTPUT VALIDATION")
    print("=" * 60)

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")
    cm = confusion_matrix(y_test, predictions)

    print(f"Accuracy : {acc:.2%}")
    print(f"F1 Score : {f1:.4f}  (weighted average)")

    print("\nConfusion Matrix:")
    print(f"{'':15}" + "".join(f"{name:>12}" for name in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:15}" + "".join(f"{val:>12}" for val in row))

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions, target_names=class_names))


def main():
    print("\n Data Classification Using AI\n")
  
    X, y, feature_names, class_names = load_and_understand_data()

   
    X_train, X_test, y_train, y_test = split_data(X, y)

    
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    
    best_k = find_best_k(X_train_scaled, y_train, X_test_scaled, y_test)

    
    model, predictions = train_and_predict(X_train_scaled, y_train, X_test_scaled, best_k)

 
    evaluate_model(y_test, predictions, class_names)

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
