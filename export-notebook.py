# %% [markdown]
# # F1 Fastest Lap Machine Learning Models

# %% [markdown]
# ## Identifying Previous F1 Meetings

# %%
import requests
import pandas as pd

url = "https://api.openf1.org/v1/meetings"
response = requests.get(url)

if response.status_code == 200:
    f1_meetings = response.json()
    print(f1_meetings)
else:
    print(f"Failed to retrieve data: {response.status_code}")



current_meeting_response = requests.get(url + "?meeting_key=latest")

if current_meeting_response.status_code == 200:
    current_meeting = current_meeting_response.json()[0]
else:
    print(f"Failed to retrieve data: {current_meeting_response.status_code}")

f1_meetings = [meeting for meeting in f1_meetings if meeting['meeting_key'] != current_meeting['meeting_key'] and "Grand Prix" in meeting['meeting_name']]

# Convert the list of meetings to a pandas DataFrame
f1_meetings_df = pd.DataFrame(f1_meetings)

print(f1_meetings_df.head())
print(f1_meetings_df.shape)


# %% [markdown]
# ## Identifying the Qualifying and Race Sessions

# %%

# Function to get all sessions for a given meeting
def get_sessions(meeting_key):
    sessions_url = f"https://api.openf1.org/v1/sessions?meeting_key={meeting_key}"
    sessions_response = requests.get(sessions_url)
    if sessions_response.status_code == 200:
        return sessions_response.json()
    return []

# Add a new column for the sessions
f1_meetings_df['sessions'] = f1_meetings_df['meeting_key'].apply(get_sessions)
print(f1_meetings_df.head())

# Function to get session keys for Race and Qualifying
def get_session_keys(sessions, session_name):
    for session in sessions:
        if session['session_name'] == session_name:
            return session['session_key']
    return None

# Add new columns for race_session_key and qualifying_session_key
f1_meetings_df['race_session_key'] = f1_meetings_df['sessions'].apply(lambda x: get_session_keys(x, 'Race'))
f1_meetings_df['qualifying_session_key'] = f1_meetings_df['sessions'].apply(lambda x: get_session_keys(x, 'Qualifying'))

print(f1_meetings_df.head())


# %% [markdown]
# ## Compiling Race Features

# %% [markdown]
# ### Race Circuit

# %%
# One-Hot encode the 'circuit_key' column
circuit_dummies = pd.get_dummies(f1_meetings_df['circuit_key'], prefix='circuit').astype(int)
f1_meetings_df = pd.concat([f1_meetings_df, circuit_dummies], axis=1)
circuit_feature_names = list(circuit_dummies.columns)
print(f1_meetings_df.head())

# %% [markdown]
# ### Year of Race

# %%
# One-Hot encode the 'year' column
year_dummies = pd.get_dummies(f1_meetings_df['year'], prefix='year').astype(int)
f1_meetings_df = pd.concat([f1_meetings_df, year_dummies], axis=1)
year_feature_names = list(year_dummies.columns)
print(f1_meetings_df.head())


# %% [markdown]
# ### Qualifying Lap Time Statistics

# %%
import requests
import pandas as pd
import statistics

# Function to get statistical information for a given session
def get_lap_statistics(session_key):
    lap_url = f"https://api.openf1.org/v1/laps?session_key={session_key}"
    lap_response = requests.get(lap_url)
    if lap_response.status_code == 200:
        laps = [lap for lap in lap_response.json() if lap['lap_duration'] is not None]
        lap_durations = [lap['lap_duration'] for lap in laps]
        
        if lap_durations:
            fastest_lap = min(lap_durations)
            average_lap = sum(lap_durations) / len(lap_durations)
            median_lap = statistics.median(lap_durations)
            std_dev = statistics.stdev(lap_durations) if len(lap_durations) > 1 else 0
            num_laps = len(lap_durations)
            
            return {
                'fastest_lap': fastest_lap,
                'average_lap': average_lap,
                'median_lap': median_lap,
                'std_dev': std_dev,
                'num_laps': num_laps
            }
    # Return None or zeros if data is unavailable
    return {
        'fastest_lap': None,
        'average_lap': None,
        'median_lap': None,
        'std_dev': None,
        'num_laps': 0
    }

# Apply the function to each qualifying session key and create a DataFrame
stats_list = f1_meetings_df['qualifying_session_key'].apply(get_lap_statistics)
stats_df = pd.DataFrame(stats_list.tolist())

# Add prefix to column names to distinguish them
stats_df = stats_df.add_prefix('qualifying_')
qualifying_feature_names = list(stats_df.columns)

# Concatenate the new statistical columns to the original DataFrame
f1_meetings_df = pd.concat([f1_meetings_df, stats_df], axis=1)

# Display the updated DataFrame
print(f1_meetings_df.head())


# %% [markdown]
# ## Race Output Variable

# %%
# Apply the function to each race session key and extract the fastest lap
f1_meetings_df['race_fastest_lap'] = f1_meetings_df['race_session_key'].apply(lambda x: get_lap_statistics(x)['fastest_lap'])

# Display the updated DataFrame
print(f1_meetings_df["race_fastest_lap"].head())


# %% [markdown]
# ## Predicting Race Fastest Laps with Multi-Layer Perceptrons

# %%
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

input_feature_names = circuit_feature_names + year_feature_names + qualifying_feature_names
output_feature_names = ["race_fastest_lap"]

print("Input Features:", input_feature_names)

# Define input features and output feature
input_features = f1_meetings_df[input_feature_names]
output_feature = f1_meetings_df[output_feature_names]

# Initialize KFold with 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mse_scores = []
r2_scores = []

mlp_model = MLPRegressor(hidden_layer_sizes=(200, 100), activation='relu', solver='adam', random_state=42)

# Perform KFold cross-validation
for train_index, test_index in kf.split(input_features):
    X_train, X_test = input_features.iloc[train_index], input_features.iloc[test_index]
    y_train, y_test = output_feature.iloc[train_index], output_feature.iloc[test_index]
    
    # Train the model
    mlp_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = mlp_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    mse_scores.append(mse)
    r2_scores.append(r2)

# Calculate average scores
average_mse = sum(mse_scores) / len(mse_scores)
average_r2 = sum(r2_scores) / len(r2_scores)

print("Average Mean Squared Error:", average_mse)
print("Average R^2 Score:", average_r2)


# %% [markdown]
# ## Evaluate Model Performances

# %% [markdown]
# ### Evaluate K-Fold Cross Validation

# %%

import matplotlib.pyplot as plt

# Plotting the results
plt.figure(figsize=(10, 5))

# Plot MSE scores
plt.subplot(1, 2, 1)
plt.plot(mse_scores, marker='o', linestyle='-', color='b')
plt.title('Mean Squared Error per Fold')
plt.xlabel('Fold')
plt.ylabel('MSE')

# Plot R^2 scores
plt.subplot(1, 2, 2)
plt.plot(r2_scores, marker='o', linestyle='-', color='r')
plt.title('R^2 Score per Fold')
plt.xlabel('Fold')
plt.ylabel('R^2')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Evaluate Feature Importancs

# %%
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Split the data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    input_features, output_feature, test_size=0.2, random_state=42
)

# Train the model
mlp_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mlp_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error on Test Set:", mse)
print("R^2 Score on Test Set:", r2)

# Output the actual and predicted fastest laps for the test set
test_results = pd.DataFrame({
    'Actual Fastest Lap': y_test.values.flatten(),
    'Predicted Fastest Lap': y_pred.flatten()
})

print("\nActual vs Predicted Fastest Laps:")
print(test_results)

# residual plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Fastest Lap')
plt.ylabel('Predicted Fastest Lap')
plt.title('Actual vs Predicted Fastest Lap')
plt.show()



# %%
# Calculate feature importances using the trained MLP model
importances = mlp_model.coefs_[0].mean(axis=1)

# Create a DataFrame to display feature importances
feature_importances = pd.DataFrame({
    'Feature': input_feature_names,
    'Importance': importances
})

# Sort the DataFrame by importance
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importances)



