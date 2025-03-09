import matplotlib.pyplot as plt
import pandas as pd

# Define constant lap time to filter around (in format mm:ss.sss)
CONSTANT_LAP_TIME = '1:18.686'

# Read lap times from CSV file
lap_data = pd.read_csv('lap_times.csv', header=None, names=['Time'])

# Convert the lap times into milliseconds
def parse_lap_times(lap_times):
    return [
        float(minutes) * 60000 + float(seconds) * 1000
        for minutes, seconds in lap_times.str.split(':').apply(lambda x: (x[0], x[1]))
    ]

lap_data['Time (ms)'] = parse_lap_times(lap_data['Time'])

# Convert constant lap time to milliseconds, including milliseconds
minutes, seconds = CONSTANT_LAP_TIME.split(':')
minutes = float(minutes)
seconds = float(seconds)
constant_time_ms = minutes * 60000 + seconds * 1000

# Filter lap times within 500 ms of the constant
filtered_lap_data = lap_data[(lap_data['Time (ms)'] >= constant_time_ms - 500) & (lap_data['Time (ms)'] <= constant_time_ms + 500)]

# Sort lap times
filtered_lap_data = filtered_lap_data.sort_values(by='Time (ms)').reset_index(drop=True)

# Add lap numbers
filtered_lap_data['Lap Number'] = range(1, len(filtered_lap_data) + 1)

# Calculate and print the average density of lap guesses within the 500ms range
average_density = len(filtered_lap_data) / 1000  # 1000 ms range (500 ms on each side)
print(f'Average density of lap guesses within 500ms range: {average_density} laps per second')

# Plotting the lap times
plt.figure(figsize=(10, 6))
plt.plot(filtered_lap_data['Lap Number'], filtered_lap_data['Time (ms)'], marker='o', linestyle='-', color='b')
plt.xlabel('Lap Number')
plt.ylabel('Time (ms)')
plt.title('Lap Times (Within 500ms of Constant)')
plt.grid(True)
plt.show()

# Removing outliers using 1.5*IQR method
Q1 = lap_data['Time (ms)'].quantile(0.25)
Q3 = lap_data['Time (ms)'].quantile(0.75)
IQR = Q3 - Q1
lap_data_no_outliers = lap_data[(lap_data['Time (ms)'] >= Q1 - 1.5 * IQR) & (lap_data['Time (ms)'] <= Q3 + 1.5 * IQR)]

# Plotting the distribution of lap times without outliers
plt.figure(figsize=(10, 6))
plt.hist(lap_data_no_outliers['Time (ms)'], bins=range(int(lap_data_no_outliers['Time (ms)'].min()), int(lap_data_no_outliers['Time (ms)'].max()) + 10, 10), color='m', edgecolor='k', alpha=0.7)
plt.xlabel('Time (ms)')
plt.ylabel('Frequency')
plt.title('Distribution of Lap Times (Without Outliers)')
plt.grid(True)
plt.show()

# Plotting the distribution of lap times with 10ms bins (Within 500ms of Constant)
plt.figure(figsize=(10, 6))
plt.hist(filtered_lap_data['Time (ms)'], bins=range(int(filtered_lap_data['Time (ms)'].min()), int(filtered_lap_data['Time (ms)'].max()) + 10, 10), color='c', edgecolor='k', alpha=0.7)
plt.xlabel('Time (ms)')
plt.ylabel('Frequency')
plt.title('Distribution of Lap Times (Within 500ms of Constant)')
plt.grid(True)
plt.show()

# Calculate differences between adjacent lap times in the filtered data
differences = filtered_lap_data['Time (ms)'].diff().dropna()

# Plotting the distribution of differences between adjacent guesses with individual bin sizes
plt.figure(figsize=(10, 6))
plt.hist(differences, bins=len(differences), color='g', edgecolor='k', alpha=0.7)
plt.xlabel('Difference between Adjacent Lap Times (ms)')
plt.ylabel('Frequency')
plt.title('Distribution of Differences Between Adjacent Lap Times (Within 500ms of Constant)')
plt.grid(True)
plt.show()

# Calculate the total time claimable by inputting 20 guesses using the top 10 differences
top_10_differences = differences.nlargest(10).sum()
total_time_claimable = top_10_differences * 2  # 20 guesses cover the top 10 differences twice
print(f'Total time claimable by inputting 20 guesses: {total_time_claimable} ms')
