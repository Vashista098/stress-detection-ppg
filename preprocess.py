import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import cheby2, filtfilt, find_peaks

# Define dataset path
dataset_folder = "D:\B21CI018\dataset\Data_29_subjects(original)\Subjects"

# Define parameters
sampling_rate = 64  # Hz
segment_length = sampling_rate * 60  # 1-minute segments (3840 samples)

# Chebyshev II Bandpass Filter Parameters
order = 4
stopband_attenuation = 20  # dB
low_cutoff = 0.5  # Hz
high_cutoff = 5.0  # Hz
nyquist = sampling_rate / 2
b, a = cheby2(order, stopband_attenuation, [low_cutoff / nyquist, high_cutoff / nyquist], btype='bandpass')

# Get the list of all subject folders
subject_folders = sorted(os.listdir(dataset_folder))

# Dictionary to store results
all_subjects_data = {}

for subject in subject_folders:
    bvp_path = os.path.join(dataset_folder, subject, "BVP.csv")
    
    if os.path.exists(bvp_path):  # Check if the file exists
        # Load BVP Data
        bvp_df = pd.read_csv(bvp_path)
        bvp_signal = bvp_df.iloc[:, 0].values  # Assuming first column is BVP
        
        # Apply Filtering
        bvp_filtered = filtfilt(b, a, bvp_signal)
        
        # Segment BVP Signal
        num_segments = len(bvp_filtered) // segment_length
        bvp_segments = [bvp_filtered[i * segment_length:(i + 1) * segment_length] for i in range(num_segments)]
        
        # Peak Detection for Each Segment
        subject_data = []
        for i, segment in enumerate(bvp_segments):
            peaks, _ = find_peaks(segment, distance=sampling_rate * 0.4)
            subject_data.append({"segment": i + 1, "num_peaks": len(peaks)})
        
        # Store results
        all_subjects_data[subject] = subject_data
        
        # Plot BVP Segment with Peaks (First 3 subjects for visualization)
        if subject_folders.index(subject) < 2:
            plt.figure(figsize=(10, 4))
            plt.plot(bvp_segments[0], label="Filtered BVP")
            plt.plot(peaks, bvp_segments[0][peaks], "ro", label="Detected Peaks")
            plt.legend()
            plt.title(f"BVP Signal with Peak Detection ({subject})")
            plt.xlabel("Samples")
            plt.ylabel("BVP Amplitude")
            plt.show()
        
        print(f"Processed {subject}: {len(bvp_segments)} segments.")

# Convert results to a DataFrame and save
results_df = pd.DataFrame.from_dict({k: [seg["num_peaks"] for seg in v] for k, v in all_subjects_data.items()}, orient='index')
results_df.to_csv("D:/b21ci018/project/bvp_peak_results.csv")

print("✅ Processing complete. Results saved to 'bvp_peak_results.csv'.")
