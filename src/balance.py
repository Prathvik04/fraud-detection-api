from imblearn.over_sampling import SMOTE

def balance_data(X, y):
    print("Before Balancing:", y.value_counts())

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    print("After Balancing:", y_res.value_counts())

    return X_res, y_res