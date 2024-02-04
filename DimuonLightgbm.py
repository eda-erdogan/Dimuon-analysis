import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# Load CSV file including 100k events
print("Loading dataset...")
df = pd.read_csv('/home/edaerdogan/Desktop/dimuonai/dimuon.csv')

signal_df = df[df['Q1'] * df['Q2'] < 0].copy()
background_df = df[df['Q1'] * df['Q2'] >= 0].copy() 

signal_df['target'] = 0
background_df['target'] = 1

filtered_df = pd.concat([signal_df, background_df], ignore_index=True)

features = filtered_df[['pt1', 'pt2', 'eta1', 'eta2', 'phi1', 'phi2', 'Q1', 'Q2']]
target = filtered_df['target']

# Split data into train and test sets
train_data, test_data, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)

# LightGBM dataset
train_dataset = lgb.Dataset(train_data, label=train_target)
test_dataset = lgb.Dataset(test_data, label=test_target, reference=train_dataset)

# LightGBM parameters
params = { #num_iterations?
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,  # Adjustable
    'learning_rate': 0.1,
    'num_trees': 32,  # Number of trees
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Custom evaluation function
def custom_eval(preds, eval_data):
    # Return a tuple (eval_name, eval_result, is_higher_better)
    return 'custom_metric', 0.0, False

'''def early_stopping_callback(env):
    current_round = env.iteration
    best_round = env.begin_iteration
    best_score = env.best_score
    is_higher_better = env.is_higher_better

    if (current_round - best_round) >= 5:  # If no improvement in the last 5 rounds
        print(f'No improvement for 5 rounds. Stopping early.')
        return True  # Stop training

    return False  '''

# Training
model = lgb.train(params,
                  train_set=train_dataset,
                  num_boost_round=100,
                  valid_sets=[train_dataset, test_dataset],
                  feval=custom_eval)
                  #early_stopping_rounds=5) #,callbacks=[early_stopping_callback])  #Early stopping issue here, no parameter defined for the algorithm?

# Predictions
train_preds = model.predict(train_data)
test_preds = model.predict(test_data)

# Convert probabilities to binary predictions
train_predictions = (train_preds >= 0.5).astype(int)
test_predictions = (test_preds >= 0.5).astype(int)

# tn, fp, fn, tp = conf_matrix.ravel() in case i need separate calc for the values

# Evaluate on the test set
accuracy = accuracy_score(test_target, test_predictions) #the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true
conf_matrix = confusion_matrix(test_target, test_predictions)
precision = precision_score(test_target, test_predictions) #tp / (tp + fp)
recall = recall_score(test_target, test_predictions) #tp / (tp + fn)
f1 = f1_score(test_target, test_predictions, average='micro') #F1 = 2 * TP / (2 * TP + FN + FP)
roc_auc = roc_auc_score(test_target, test_preds) #Area Under the Receiver Operating Characteristic Curve

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC AUC: {roc_auc}')
