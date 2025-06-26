import os
import glob
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def find_latest_file(pattern):
    try:
        return max(glob.glob(pattern), key=os.path.getctime)
    except ValueError:
        return None

latest_log_file = find_latest_file('training_log_*.csv')

if not latest_log_file:
    print("Could not find a training log")
    exit()

print(f"Analyzing training log: {os.path.basename(latest_log_file)}")

# plotting
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

try:
    log_df = pd.read_csv(latest_log_file)
    # calculate win rate 
    log_df['win'] = (log_df['outcome'] == 'win').astype(int)
    win_rate_rolling = log_df['win'].rolling(window=100, min_periods=10).mean() # use a 10 period min to smooth the start
    
    ax.plot(log_df['episode'], win_rate_rolling, label='Win Rate (100-episode rolling average)', color='royalblue', linewidth=2)
    ax.set_title('Agent Learning Progress During Training', fontsize=16, pad=20)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Win Rate', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10)
    
except FileNotFoundError:
    ax.text(0.5, 0.5, f'Log file not found:\n{os.path.basename(latest_log_file)}', ha='center', va='center', fontsize=12)
    ax.set_title('Learning Progress Analysis', fontsize=16)
except Exception as e:
    ax.text(0.5, 0.5, f'An error occurred while reading the log file:\n{e}', ha='center', va='center', fontsize=12)
    ax.set_title('Learning Progress Analysis', fontsize=16)

plt.tight_layout()


timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
output_filename = f"evaluation_log_analysis_{timestamp}.png"
plt.savefig(output_filename)
print(f"\nTraining log analysis plot saved to: {output_filename}")
plt.close() 