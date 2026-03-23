'''
HELPER FUNCTIONS FOR DATA CLEANING 
PHASE 1 & PHASE 2

Contains any re-used data cleaning functions between phase 1 & phase 2. 
'''

import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

##############################################################################
def load_and_merge(
                        ROOT_DIR, 
                        CONDITION_MAP, 
                        FS = 10,
                        CUTOFF = 0.1,
                        ORDER = 4
                   ):
    
    '''
    LOADING & MERGING DATA
    '''
    
    # setup the low-pass filter
    b, a = butter(ORDER, CUTOFF / (0.5 * FS), btype="low")

    all_data = []

    # looping through all files in Phase 1 of LOGS
    for root, dirs, files in os.walk(ROOT_DIR):

        for file in files:

            filepath = os.path.join(root, file)

            # excluding MacOS's inevitable log 
            if file == ".DS_Store":
                continue

            else:

                # determining condition from folder name
                condition = None
                for folder_name in CONDITION_MAP:

                    if folder_name in root:
                        condition = CONDITION_MAP[folder_name]
                        break

                if condition is None:
                    continue

                df = pd.read_csv(filepath)

                # ---- SORT BY TIME FIRST ----
                df = df.sort_values("timestamp").copy()

                # ---- APPLY FILTERS ----
                df["Power_W_raw"] = df["Power_W"]
                df["MM_Magnitude_raw"] = df["MM_Magnitude"]

                # low-pass for power
                if len(df) > max(len(a), len(b)) * 3:
                    df["Power_W"] = filtfilt(b, a, df["Power_W"])   # aggressive (0.1 Hz)

                # Savitzky-Golay for mm
                df["MM_Magnitude"] = savgol_filter(
                                                    df["MM_Magnitude"],
                                                    window_length=51,   # must be odd
                                                    polyorder=2
                                                  )

                # ---- Reset time per file ----
                df["time_sec"] = (df["timestamp"] - df["timestamp"].iloc[0])

                # ---- Trim first 5 seconds ----
                df = df[df["time_sec"] >= 5].reset_index(drop=True)

                # ---- Add metadata AFTER trimming ----
                df["Condition"] = condition
                df["Trial_File"] = file

                all_data.append(df)

    return pd.concat(all_data, ignore_index=True)
##############################################################################
def nominal_offset_from_filename(trial_file):
    """
    Extract nominal offset class from filename.
    Examples:
        0mm_1.csv -> 0.0
        1mm_4.csv -> 1.0
        2mm_2.csv -> 2.0
    """
    name = str(trial_file)
    if name.startswith("0mm"):
        return 0.0
    elif name.startswith("1mm"):
        return 1.0
    elif name.startswith("2mm"):
        return 2.0
    
    return np.nan
##############################################################################
def build_mm_corrected(
                            group,
                            min_time=8.0,
                            min_spacing_sec=3.0,
                            prominence_power=None,
                            top_n_peaks=3,
                            apply_nominal_floor=True
                        ):
    """
    Correct MM_Magnitude using FBG-trough-locked anchoring.

    Logic:
    1. find the top N strongest FBG troughs in the file
    2. read the MM value at each exact FBG trough index
    3. choose the lowest of those trough-aligned MM values
    4. shift the whole file so that chosen value = nominal offset
    5. optionally apply a hard floor at the nominal offset
    """
    group = group.sort_values("time_sec").copy()

    nominal_offset = nominal_offset_from_filename(group["Trial_File"].iloc[0])

    # fallback if nominal can't be read
    if pd.isna(nominal_offset):
        group["MM_corrected"] = group["MM_Magnitude"]
        group["Nominal_Offset_mm"] = nominal_offset
        group["MM_anchor_value"] = np.nan
        group["MM_anchor_time_sec"] = np.nan
        group["MM_floor_applied"] = False
        return group

    work = group[group["time_sec"] >= min_time].copy()

    # fallback if too little data
    if len(work) < 20:
        mm_min = group["MM_Magnitude"].min()
        mm_corrected = group["MM_Magnitude"] - mm_min + nominal_offset
        if apply_nominal_floor:
            mm_corrected = np.maximum(mm_corrected, nominal_offset)

        group["MM_corrected"] = mm_corrected
        group["Nominal_Offset_mm"] = nominal_offset
        group["MM_anchor_value"] = mm_min
        group["MM_anchor_time_sec"] = group.loc[group["MM_Magnitude"].idxmin(), "time_sec"]
        group["MM_floor_applied"] = apply_nominal_floor
        return group

    t = work["time_sec"].to_numpy()
    y_pow = work["Power_denoised"].to_numpy(dtype=float)
    y_mm  = work["MM_Magnitude"].to_numpy(dtype=float)

    dt = np.median(np.diff(t)) if len(t) > 1 else 0.1
    distance_samples = max(1, int(min_spacing_sec / dt))

    # -------------------------
    # Find FBG troughs
    # -------------------------
    yrange = np.nanmax(y_pow) - np.nanmin(y_pow)
    if prominence_power is None:
        prominence_power = 0.06 * yrange if yrange > 0 else 0.0

    trough_idx, _ = find_peaks(
                                -y_pow,
                                distance=max(1, int(0.75 * distance_samples)),
                                prominence=prominence_power
                            )

    # fallback if no troughs found
    if len(trough_idx) == 0:
        mm_min = group["MM_Magnitude"].min()
        mm_corrected = group["MM_Magnitude"] - mm_min + nominal_offset
        if apply_nominal_floor:
            mm_corrected = np.maximum(mm_corrected, nominal_offset)

        group["MM_corrected"] = mm_corrected
        group["Nominal_Offset_mm"] = nominal_offset
        group["MM_anchor_value"] = mm_min
        group["MM_anchor_time_sec"] = group.loc[group["MM_Magnitude"].idxmin(), "time_sec"]
        group["MM_floor_applied"] = apply_nominal_floor
        return group

    # keep only top N strongest troughs (deepest minima)
    trough_idx = trough_idx[np.argsort(y_pow[trough_idx])]
    trough_idx = trough_idx[:top_n_peaks]
    trough_idx = np.sort(trough_idx)

    # -------------------------
    # Use MM value at the exact FBG trough index
    # -------------------------
    candidate_mm_at_troughs = []

    for p_idx in trough_idx:
        mm_at_trough = y_mm[p_idx]
        time_at_trough = t[p_idx]

        if np.isnan(mm_at_trough):
            continue

        candidate_mm_at_troughs.append((mm_at_trough, time_at_trough))

    # fallback if all trough-matched MM values are invalid
    if len(candidate_mm_at_troughs) == 0:
        mm_min = group["MM_Magnitude"].min()
        mm_corrected = group["MM_Magnitude"] - mm_min + nominal_offset
        if apply_nominal_floor:
            mm_corrected = np.maximum(mm_corrected, nominal_offset)

        group["MM_corrected"] = mm_corrected
        group["Nominal_Offset_mm"] = nominal_offset
        group["MM_anchor_value"] = mm_min
        group["MM_anchor_time_sec"] = group.loc[group["MM_Magnitude"].idxmin(), "time_sec"]
        group["MM_floor_applied"] = apply_nominal_floor
        return group

    # choose the lowest MM value among the exact trough-aligned samples
    anchor_mm_value, anchor_time_sec = min(candidate_mm_at_troughs, key=lambda x: x[0])

    # shift full file so anchor becomes nominal
    mm_corrected = group["MM_Magnitude"] - anchor_mm_value + nominal_offset

    # hard floor at nominal
    if apply_nominal_floor:
        mm_corrected = np.maximum(mm_corrected, nominal_offset)

    group["MM_corrected"] = mm_corrected
    group["Nominal_Offset_mm"] = nominal_offset
    group["MM_anchor_value"] = anchor_mm_value
    group["MM_anchor_time_sec"] = anchor_time_sec
    group["MM_floor_applied"] = apply_nominal_floor

    return group
##############################################################################
def choose_event_center_index(y_mm, y_pow, min_idx, trough_idx, ratio_weight_mm=0.5, ratio_weight_pow=0.5):
    """
    Event center is chosen using a ratio-like score between:
    - closeness to MM minimum
    - closeness to FBG minimum (power trough)

    If multiple equal MM minima exist, the one closest to the FBG trough is favored.
    """
    mm_min_val = np.nanmin(y_mm)
    pow_min_val = np.nanmin(y_pow)

    # candidate region between mm minimum and power trough
    left = min(min_idx, trough_idx)
    right = max(min_idx, trough_idx)

    idxs = np.arange(left, right + 1)

    mm_range = np.nanmax(y_mm) - np.nanmin(y_mm)
    pow_range = np.nanmax(y_pow) - np.nanmin(y_pow)

    if mm_range == 0:
        mm_score = np.ones(len(idxs))
    else:
        # higher score when closer to mm minimum
        mm_score = 1 - ((y_mm[idxs] - mm_min_val) / mm_range)

    if pow_range == 0:
        pow_score = np.ones(len(idxs))
    else:
        # higher score when closer to power minimum
        pow_score = 1 - ((y_pow[idxs] - pow_min_val) / pow_range)

    combined_score = ratio_weight_mm * mm_score + ratio_weight_pow * pow_score

    best_local = idxs[np.argmax(combined_score)]
    
    return best_local
##############################################################################
def find_event_end_from_power(y_pow, start_idx):
    """
    Starting from the event center/trough, move right until power begins decreasing again.
    This keeps the event frame through the recovery side of the trough.

    We stop at the first sustained decrease.
    """
    if start_idx >= len(y_pow) - 2:
        return len(y_pow) - 1

    for i in range(start_idx + 1, len(y_pow) - 1):
        # stop once the recovering signal turns downward again
        if y_pow[i + 1] < y_pow[i]:
            return i

    return len(y_pow) - 1
##############################################################################
def is_complete_peak(y_pow, trough_idx, edge_buffer=3):
    """
    Accept a trough if:
    - it is not too close to the edges
    - it has a falling side before the trough
    - it has a rising side after the trough
    """
    if trough_idx < edge_buffer or trough_idx > len(y_pow) - 1 - edge_buffer:
        return False

    left = y_pow[trough_idx - edge_buffer:trough_idx]
    right = y_pow[trough_idx + 1:trough_idx + 1 + edge_buffer]

    if len(left) < 2 or len(right) < 2:
        return False

    left_falls = np.any(np.diff(left) < 0)
    right_rises = np.any(np.diff(right) > 0)

    return left_falls and right_rises
##############################################################################
def extract_events_from_trial(
                                group,
                                min_time=8.0,
                                min_spacing_sec=4.0,
                                prominence_mm=1.0,
                                prominence_power=None,
                                mm_peak_match_window_sec=3.0,
                                pre_sec=1.5,
                                post_sec=3.0
                            ):
    """
    Event logic:
    1. find MM minima candidates
    2. find FBG trough candidates
    3. pair each MM minimum with a nearby FBG trough
    4. if MM minimum is repeated/flat, use the one closest to FBG trough
    5. define event center using a combined ratio score between MM minimum and FBG trough
    6. event starts pre_sec before center
    7. event ends when FBG starts decreasing again after recovery
    8. reject incomplete/chopped/non-trough events immediately
    """

    group = group.sort_values("time_sec").copy()
    work = group[group["time_sec"] >= min_time].copy()

    if len(work) < 20:
        return pd.DataFrame()

    t = work["time_sec"].to_numpy()
    y_mm = work["MM_corrected"].to_numpy(dtype=float)
    y_pow = work["Power_denoised"].to_numpy(dtype=float)

    dt = np.median(np.diff(t)) if len(t) > 1 else 0.1
    distance_samples = max(1, int(min_spacing_sec / dt))
    match_window_samples = max(1, int(mm_peak_match_window_sec / dt))

    # MM minima
    minima_idx, _ = find_peaks(
                                    -y_mm,
                                    distance=distance_samples,
                                    prominence=prominence_mm
                                )

    if len(minima_idx) == 0:
        return pd.DataFrame()

    # FBG troughs
    yrange = np.nanmax(y_pow) - np.nanmin(y_pow)
    if prominence_power is None:
        prominence_power = 0.10 * yrange if yrange > 0 else 0.0

    trough_idx, _ = find_peaks(
                                    -y_pow,
                                    distance=max(1, int(0.75 * distance_samples)),
                                    prominence=prominence_power
                                )

    if len(trough_idx) == 0:
        return pd.DataFrame()

    # -------------------------
    # Keep only the top 3 deepest power troughs in this file
    # -------------------------
    top_n_peaks = 3

    # sort candidate troughs by power depth, lowest first
    trough_idx = trough_idx[np.argsort(y_pow[trough_idx])]

    # keep only the strongest top_n_peaks troughs
    trough_idx = trough_idx[:top_n_peaks]

    # re-sort into time order so event numbering follows the file timeline
    trough_idx = np.sort(trough_idx)

    rows = []

    for event_num, p_idx in enumerate(trough_idx, start=1):

        nearby_troughs = trough_idx[np.abs(trough_idx - p_idx) <= match_window_samples]
        if len(nearby_troughs) == 0:
            continue

        # choose deepest nearby FBG trough
        best_trough_idx = nearby_troughs[np.argmin(y_pow[nearby_troughs])]

        # handle repeated / flat MM minima:
        # choose the minimum point closest to the selected FBG trough
        local_left = max(0, p_idx - max(1, int(0.5 / dt)))
        local_right = min(len(y_mm), p_idx + max(1, int(0.5 / dt)) + 1)

        local_mm = y_mm[local_left:local_right]
        local_idxs = np.arange(local_left, local_right)

        local_min_val = np.nanmin(local_mm)
        tol = 1e-9
        repeated_min_idxs = local_idxs[np.abs(local_mm - local_min_val) <= tol]

        if len(repeated_min_idxs) > 1:
            chosen_min_idx = repeated_min_idxs[np.argmin(np.abs(repeated_min_idxs - best_trough_idx))]
        else:
            chosen_min_idx = int(repeated_min_idxs[0])

        # center based on combined MM-min / FBG-trough ratio score
        center_idx = choose_event_center_index(
                                                y_mm=y_mm,
                                                y_pow=y_pow,
                                                min_idx=chosen_min_idx,
                                                trough_idx=best_trough_idx,
                                                ratio_weight_mm=0.5,
                                                ratio_weight_pow=0.5
                                            )

        # define start/end
        '''
        start_time = t[center_idx] - pre_sec
        end_idx = find_event_end_from_power(y_pow, best_trough_idx)
        end_time = t[end_idx]
        '''
        
        start_idx = max(0, int(center_idx - round(pre_sec / dt)))
        end_idx   = min(len(t) - 1, int(center_idx + post_sec / dt))

        start_time = t[start_idx]
        end_time   = t[end_idx]

        # reject chopped or backward windows
        if end_time <= start_time:
            continue

        # require true complete trough
        if not is_complete_peak(y_pow, best_trough_idx, edge_buffer=3):
            continue

        event_df = work[
                            (work["time_sec"] >= start_time) &
                            (work["time_sec"] <= end_time)
                        ].copy()

        if len(event_df) < 8:
            continue

        # require event to actually contain the selected power trough
        if not ((event_df["time_sec"].min() <= t[best_trough_idx]) and (t[best_trough_idx] <= event_df["time_sec"].max())):
            continue

        power = event_df["Power_denoised"].to_numpy(dtype=float)
        mm = event_df["MM_corrected"].to_numpy(dtype=float)
        event_t = event_df["time_sec"].to_numpy(dtype=float)

        power_diff = np.diff(power) if len(power) > 1 else np.array([0.0])

        row = {
            "Trial_File": group["Trial_File"].iloc[0],
            "Event_ID": f"{group['Trial_File'].iloc[0]}__event_{event_num}",
            "Nominal_Class": group["Trial_File"].iloc[0].split("_")[0],

            "mm_min_time_sec": t[chosen_min_idx],
            "power_trough_time_sec": t[best_trough_idx],
            "event_center_time_sec": t[center_idx],
            "event_start_time_sec": start_time,
            "event_end_time_sec": end_time,

            "event_duration_sec": end_time - start_time,
            "mm_to_trough_dt_sec": t[best_trough_idx] - t[chosen_min_idx],
            "center_to_trough_dt_sec": t[best_trough_idx] - t[center_idx],

            "mm_min_value": np.nanmin(mm),
            "mm_median": np.nanmedian(mm),
            "mm_mean": np.nanmean(mm),

            "power_mean": np.nanmean(power),
            "power_std": np.nanstd(power),
            "power_min": np.nanmin(power),
            "power_max": np.nanmax(power),
            "power_ptp": np.nanmax(power) - np.nanmin(power),
            "power_median": np.nanmedian(power),
            "power_energy": np.trapz(power ** 2, event_t) if len(event_t) >= 2 else np.nansum(power ** 2),
            "power_rms": np.sqrt(np.nanmean(power ** 2)),
            "power_max_slope": np.nanmax(np.abs(power_diff)) if len(power_diff) else 0.0,
            "power_slope_event": np.polyfit(event_t, power, 1)[0] if len(event_t) >= 2 else 0.0,
            "power_curvature": np.nanmean(np.diff(power, 2)) if len(power) >= 3 else 0.0,

            "n_samples": len(event_df),
        }

        rows.append(row)

    return pd.DataFrame(rows)
##############################################################################
import matplotlib.pyplot as plt

def plot_before_after_file(
    df,
    trial_file,
    time_col="time_sec",
    mm_raw_col="MM_Magnitude_raw",
    mm_filtered_col="MM_Magnitude",
    mm_final_col="MM_corrected",
    power_raw_col="Power_W_raw",
    power_filtered_col="Power_denoised",
    power_final_col="Power_W",
    show_events=False
):
    """
    Plot one file showing:
    1) MM error: raw vs filtered vs final
    2) FBG power: raw vs filtered vs final

    Assumes df already contains one row per timestamp and includes Trial_File.
    """

    g = df[df["Trial_File"] == trial_file].copy().sort_values(time_col)

    if g.empty:
        print(f"No rows found for {trial_file}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # -------------------------
    # MM plot
    # -------------------------
    if mm_raw_col in g.columns:
        axes[0].plot(g[time_col], g[mm_raw_col], label="MM raw", alpha=0.6)
    if mm_filtered_col in g.columns:
        axes[0].plot(g[time_col], g[mm_filtered_col], label="MM filtered", linewidth=2)
    if mm_final_col in g.columns:
        axes[0].plot(g[time_col], g[mm_final_col], label="MM final/corrected", linewidth=2)

    # optional anchor line
    if "Nominal_Offset_mm" in g.columns and g["Nominal_Offset_mm"].notna().any():
        nominal = g["Nominal_Offset_mm"].dropna().iloc[0]
        axes[0].axhline(nominal, linestyle="--", alpha=0.7, label=f"Nominal = {nominal:.1f} mm")

    # optional anchor time
    if "MM_anchor_time_sec" in g.columns and g["MM_anchor_time_sec"].notna().any():
        anchor_t = g["MM_anchor_time_sec"].dropna().iloc[0]
        axes[0].axvline(anchor_t, linestyle=":", alpha=0.7, label="MM anchor")

    axes[0].set_title(f"{trial_file} — MM Error Before/After")
    axes[0].set_ylabel("MM Error")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # -------------------------
    # Power plot
    # -------------------------
    if power_raw_col in g.columns:
        axes[1].plot(g[time_col], g[power_raw_col], label="Power raw", alpha=0.6)
    if power_filtered_col in g.columns:
        axes[1].plot(g[time_col], g[power_filtered_col], label="Power filtered", linewidth=2)
    if power_final_col in g.columns:
        axes[1].plot(g[time_col], g[power_final_col], label="Power final/denoised", linewidth=2)

    axes[1].set_title(f"{trial_file} — FBG Power Before/After")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Power")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # -------------------------
    # Optional event markers
    # -------------------------
    if show_events and "event_center_time_sec" in g.columns:
        for ax in axes:
            for t in g["event_center_time_sec"].dropna().unique():
                ax.axvline(t, linestyle="--", alpha=0.25)

    plt.tight_layout()
    plt.show()
##############################################################################