import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import cheby2, filtfilt, find_peaks, welch
from scipy.stats import skew, kurtosis

# Define dataset path
dataset_folder = "D:/B21CI018/dataset/Data_29_subjects(original)/Subjects"

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

# Dictionary to store features
all_features = []

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
        
        # Extract Features from Each Segment
        for i, segment in enumerate(bvp_segments):
            peaks, _ = find_peaks(segment, distance=sampling_rate * 0.4)
            peak_intervals = np.diff(peaks) / sampling_rate  # Convert to seconds
            
            # Remove abnormal intervals (< 500ms or > 1200ms)
            valid_intervals = peak_intervals[(peak_intervals >= 0.5) & (peak_intervals <= 1.2)]
            
            # If more than 15% of the segment is abnormal, discard it
            if len(valid_intervals) < 0.85 * len(peak_intervals):
                continue
            
            # Time-Domain Features
            mean_val = np.mean(segment)
            variance_val = np.var(segment)
            skewness_val = skew(segment)
            kurtosis_val = kurtosis(segment)
            peak_to_peak = np.ptp(segment)
            
            # Frequency-Domain Features
            freqs, psd = welch(segment, fs=sampling_rate)
            dominant_freq = freqs[np.argmax(psd)]  # Frequency with highest power
            
            # Spectral Features
            spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
            spectral_bandwidth = np.sqrt(np.sum((freqs - spectral_centroid)**2 * psd) / np.sum(psd))
            spectral_contrast = np.max(psd) - np.min(psd)
            spectral_flatness = np.exp(np.mean(np.log(psd))) / np.mean(psd)
            spectral_rolloff = freqs[np.where(np.cumsum(psd) >= 0.9 * np.sum(psd))[0][0]]
            
            # Labeling Based on HR Variability (Threshold from Paper)
            avg_hr = 60 / np.mean(valid_intervals)  # Convert interval to BPM
            label = 1 if avg_hr > 90 else 0  # Stress if HR > 90 BPM
            
            # Store Features
            all_features.append([
                subject, i + 1, mean_val, variance_val, skewness_val, kurtosis_val, peak_to_peak,
                dominant_freq, spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_flatness, spectral_rolloff, avg_hr, label
            ])
        
        print(f"Extracted features for {subject}: {len(bvp_segments)} segments.")

# Save features to CSV
feature_columns = [
    "Subject", "Segment", "Mean", "Variance", "Skewness", "Kurtosis", "Peak-to-Peak Amplitude",
    "Dominant Frequency", "Spectral Centroid", "Spectral Bandwidth", "Spectral Contrast", "Spectral Flatness", "Spectral Roll-off", "Avg_HR", "Label"
]
features_df = pd.DataFrame(all_features, columns=feature_columns)
features_df.to_csv("D:/b21ci018/project/bvp_features.csv", index=False)

print("✅ Feature extraction with spectral features complete. Results saved to 'bvp_features.csv'.")
