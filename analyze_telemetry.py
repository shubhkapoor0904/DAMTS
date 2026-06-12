import csv
import math
import os
import sys
import re

def calculate_percentile(data, percentile):
    if not data:
        return None
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * percentile
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data_sorted[int(k)]
    d0 = data_sorted[int(f)] * (c - k)
    d1 = data_sorted[int(c)] * (k - f)
    return d0 + d1

def main():
    csv_file = "phone_confidence_telemetry.csv"
    if not os.path.exists(csv_file):
        print(f"\nError: Telemetry file '{csv_file}' not found.")
        print("Please follow these steps:")
        print("  1. Run the system: python integrated_dms.py")
        print("  2. Hold your phone in natural usage positions for 5-10 seconds during monitoring.")
        print("  3. Exit the application by pressing the ESC key.")
        print("  4. Run this analysis script again to tune the threshold.")
        sys.exit(1)

    confidences = []
    
    with open(csv_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                conf = float(row['confidence'])
                confidences.append(conf)
            except (ValueError, KeyError):
                continue

    if not confidences:
        print(f"\nNo phone telemetry data found in '{csv_file}'.")
        print("Please ensure you hold a phone in view of the camera during the monitoring phase (after calibration is done) to generate telemetry data.")
        sys.exit(0)

    total_samples = len(confidences)
    mean_conf = sum(confidences) / total_samples
    min_conf = min(confidences)
    max_conf = max(confidences)
    
    p10 = calculate_percentile(confidences, 0.10)
    p25 = calculate_percentile(confidences, 0.25)
    p50 = calculate_percentile(confidences, 0.50) # Median
    p75 = calculate_percentile(confidences, 0.75)
    p90 = calculate_percentile(confidences, 0.90)

    variance = sum((x - mean_conf) ** 2 for x in confidences) / total_samples
    std_dev = math.sqrt(variance)

    print("\n" + "=" * 60)
    print("         DAMTS PHONE DETECTION TELEMETRY ANALYSIS         ")
    print("=" * 60)
    print(f"Total Telemetry Frames logged : {total_samples}")
    print(f"Confidence Range              : {min_conf:.3f} to {max_conf:.3f}")
    print(f"Mean Confidence               : {mean_conf:.3f} (StdDev: {std_dev:.3f})")
    print("-" * 60)
    print("Percentile Distribution:")
    print(f"  10th Percentile (P10)       : {p10:.3f} (90% of frames were above this)")
    print(f"  25th Percentile (P25)       : {p25:.3f} (75% of frames were above this)")
    print(f"  50th Percentile (P50/Median): {p50:.3f}")
    print(f"  75th Percentile (P75)       : {p75:.3f}")
    print(f"  90th Percentile (P90)       : {p90:.3f}")
    print("=" * 60)

    # ASCII Histogram
    print("\nConfidence Score Distribution Histogram:")
    bins = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
    bin_counts = [0] * (len(bins) - 1)
    
    for conf in confidences:
        for i in range(len(bins) - 1):
            if bins[i] <= conf < bins[i+1]:
                bin_counts[i] += 1
                break

    max_count = max(bin_counts) if bin_counts else 0
    max_bar_width = 30
    
    for i in range(len(bin_counts)):
        bin_label = f"[{bins[i]:.2f} - {bins[i+1]:.2f})"
        count = bin_counts[i]
        percentage = (count / total_samples) * 100
        bar_len = int((count / max_count) * max_bar_width) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  {bin_label:<15} : {bar:<30} {count:>4} ({percentage:.1f}%)")

    print("=" * 60)

    # Recommendation
    # p10 is a good baseline because it will capture 90% of active frames, 
    # but we clip it to ensure it is not too low (causing false alarms) or too high.
    recommended_threshold = max(0.18, min(0.40, p10 if p10 else 0.25))
    recommended_threshold = round(recommended_threshold, 2)
    
    print(f"\nRecommended Phone Confidence Threshold (YOLO_CONF_PHONE): {recommended_threshold:.2f}")
    print("  (This represents your P10 score, ensuring 90% detection coverage of phone frames)")

    try:
        response = input(f"\nDo you want to update integrated_dms.py to use this threshold ({recommended_threshold:.2f})? (y/n): ").strip().lower()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)
        
    if response == 'y':
        dms_file = "integrated_dms.py"
        if os.path.exists(dms_file):
            with open(dms_file, "r") as f:
                content = f.read()
            
            pattern = r"(YOLO_CONF_PHONE_CONFIRMED\s*=\s*)[0-9\.]+"
            new_content, count = re.subn(pattern, f"\\g<1>{recommended_threshold:.2f}", content)
            
            if count > 0:
                with open(dms_file, "w") as f:
                    f.write(new_content)
                print(f"\nSuccessfully updated YOLO_CONF_PHONE_CONFIRMED to {recommended_threshold:.2f} in integrated_dms.py!")
            else:
                print("\nError: Could not locate YOLO_CONF_PHONE_CONFIRMED variable in integrated_dms.py.")
        else:
            print(f"\nError: '{dms_file}' not found in the current directory.")
    else:
        print("\nTuning aborted. No files were modified.")

if __name__ == "__main__":
    main()
