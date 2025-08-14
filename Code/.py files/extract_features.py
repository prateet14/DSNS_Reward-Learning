import pandas as pd
import numpy as np
import os
import glob
import re

def extract_trial_date(filename):
    match = re.search(r"(\d{2}-\d{2}-\d{4})", filename)
    return match.group(1) if match else "Unknown"

def extract_animal_name(filename):
    return filename.split("_")[1]

def trials_type(df):
    trial_name_start = []
    trial_name_end = []
    for column in df.columns:
        if "Trial name" in column:
            if "started" in column:
                trial_name_start.append(column)
            elif "Finished" in column:
                trial_name_end.append(column)
    return trial_name_start, trial_name_end

def process_day_folder(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    summary_data = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        animal_name = extract_animal_name(os.path.basename(csv_file))

        control_rate, bin1_rate, bin2_rate, bin3_rate, bin4_rate, bin5_rate, bin6_rate, bin7_rate, bin8_rate, bin9_rate, bin10_rate, prediction_rate, reward_rate = ([] for _ in range(13))

        trial_starts, trial_ends = trials_type(df)
        if not trial_starts or not trial_ends:
            continue

        start_col = trial_starts[0]
        end_col = trial_ends[0]

        for i in range(200):
            start_trial = df.loc[i, start_col]
            end_trial = df.loc[i, end_col]

            if pd.isna(start_trial) or pd.isna(end_trial):
                continue

            reward_values = df.loc[(df['Dev1/port0/line7 S output'] > start_trial) &
                                   (df['Dev1/port0/line7 S output'] < end_trial), 'Dev1/port0/line7 S output']

            tone_values = df.loc[(df['Tone S output'] > start_trial) &
                                 (df['Tone S output'] < end_trial), 'Tone S output']

            if reward_values.empty or tone_values.empty:
                continue

            reward_onset = reward_values.iloc[0]
            tone_onset = tone_values.iloc[0]

            starting_point = tone_onset - 2.49
            controlbin = tone_onset - 2.24
            bin1 = tone_onset - 1.99
            bin2 = tone_onset - 1.74
            bin3 = tone_onset - 1.49
            bin4 = tone_onset - 1.24
            bin5 = tone_onset - 0.99
            bin6 = tone_onset - 0.74
            bin7 = tone_onset - 0.49
            bin8 = tone_onset - 0.24
            bin9 = tone_onset + 0.01
            predbin = reward_onset - 0.24
            rewardbin = reward_onset + 0.01
            finishbin = reward_onset + 0.26

            licking = df['Dev1/ai13 L input']
            control_rate.append(licking[(licking > starting_point) & (licking < controlbin)].count())
            bin1_rate.append(licking[(licking >= controlbin) & (licking < bin1)].count())
            bin2_rate.append(licking[(licking >= bin1) & (licking < bin2)].count())
            bin3_rate.append(licking[(licking >= bin2) & (licking < bin3)].count())
            bin4_rate.append(licking[(licking >= bin3) & (licking < bin4)].count())
            bin5_rate.append(licking[(licking >= bin4) & (licking < bin5)].count())
            bin6_rate.append(licking[(licking >= bin5) & (licking < bin6)].count())
            bin7_rate.append(licking[(licking >= bin6) & (licking < bin7)].count())
            bin8_rate.append(licking[(licking >= bin7) & (licking < bin8)].count())
            bin9_rate.append(licking[(licking >= bin8) & (licking < bin9)].count())
            bin10_rate.append(licking[(licking >= bin9) & (licking < predbin)].count())
            prediction_rate.append(licking[(licking >= predbin) & (licking < rewardbin)].count())
            reward_rate.append(licking[(licking >= rewardbin) & (licking < finishbin)].count())

        if bin1_rate:
            bin_means = [np.mean(b) for b in [
                control_rate, bin1_rate, bin2_rate, bin3_rate, bin4_rate,
                bin5_rate, bin6_rate, bin7_rate, bin8_rate, bin9_rate,
                bin10_rate, prediction_rate, reward_rate
            ]]
            summary_data.append({
                'Animal': animal_name,
                'Control': bin_means[0],
                'Bin1': bin_means[1],
                'Bin2': bin_means[2],
                'Bin3': bin_means[3],
                'Bin4': bin_means[4],
                'Bin5': bin_means[5],
                'Bin6': bin_means[6],
                'Bin7': bin_means[7],
                'Bin8': bin_means[8],
                'Bin9': bin_means[9],
                'Bin10': bin_means[10],
                'Prediction': bin_means[11],
                'Reward': bin_means[12]
            })

    return pd.DataFrame(summary_data)

def main():
    base_folder_path = r"C:\Users\Dell\Desktop\Data Science Applications in Neuroscience\Dataset"
    day_folders = ["day 1", "day 2", "day 3", "day 4", "day 5"]

    processed_folder = os.path.join(base_folder_path, "Features Table")
    os.makedirs(processed_folder, exist_ok=True)

    for day_name in day_folders:
        folder_path = os.path.join(base_folder_path, day_name)
        summary_df = process_day_folder(folder_path)
        output_path = os.path.join(processed_folder, f"features_{day_name.replace(' ', '')}.csv")
        summary_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()