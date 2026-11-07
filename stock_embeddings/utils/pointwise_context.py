from typing import List, Tuple

import numpy as np
from pandas import DataFrame
from tqdm import tqdm

def get_pointwise_tgt_context_sets(
    df: DataFrame,
    context_size: int = 4,
    verbose: bool = False,
    iqr_noise_reduction: bool = True,
    drop_nans: bool = False,
) -> List[Tuple[int, List[int]]]:
    """
    Generate (target_idx, [context_idxs]) pairs by sliding a window over the
    sorted returns of each day, with edge padding so the extreme tickers also get a full context.
    Also report how many days were skipped for having too few valid tickers,
    and how many ticker-positions were dropped due to NaNs.

    Args:
        df: returns DataFrame (rows = days, cols = tickers)
        context_size: must be even; window size = context_size + 1
        verbose: show progress bar and summary
        iqr_noise_reduction: if True, skip windows whose center falls in the
            middle 50% of the sorted list (to reduce noise)
        drop_nans: if True, exclude any ticker with NaN on that day

    Returns:
        List of (target_idx, context_idx_list) for all days and all valid windows.
    """
    if context_size % 2 != 0:
        raise ValueError(f"Context size must be even, got {context_size}")

    arr = df.values  # shape = (n_days, n_tickers)
    n_days, n_ts = arr.shape
    half = context_size // 2

    out: List[Tuple[int, List[int]]] = []
    days_skipped = 0
    tickers_dropped = 0

    iterator = range(n_days)
    if verbose:
        iterator = tqdm(iterator, desc="Days")

    for day in iterator:
        day_vals = arr[day, :]

        # 1) choose tickers to consider
        if drop_nans:
            valid = np.where(~np.isnan(day_vals))[0]
            tickers_dropped += (n_ts - len(valid))
        else:
            valid = np.arange(n_ts)

        # 2) need at least context_size + 1 tickers
        if len(valid) < context_size + 1:
            days_skipped += 1
            continue

        # 3) sort those tickers by return
        sorted_idx = valid[np.argsort(day_vals[valid])]
        m = len(sorted_idx)

        # 4) edge-pad so the first and last tickers get a full window
        front_pad = sorted_idx[half + 1 : context_size + 1]
        back_pad  = sorted_idx[-context_size - 1 : -half - 1]
        padded = np.concatenate([front_pad, sorted_idx, back_pad])

        # 5) determine which window centers to keep
        positions = list(range(m))
        if iqr_noise_reduction:
            start_i = int(m * 0.25)
            end_i   = int(m * 0.75)
            positions = [i for i in positions if not (start_i < i < end_i)]

        # 6) slide the window over padded array
        for pos in positions:
            center = pos + half
            window = padded[center - half : center + half + 1]  # length = context_size + 1
            tgt = int(window[half])
            ctx = [int(x) for i, x in enumerate(window) if i != half]
            out.append((tgt, ctx))

    if verbose:
        print(f"→ Generated {len(out)} target–context pairs "
              f"(drop_nans={drop_nans}, iqr_noise_reduction={iqr_noise_reduction}).")
        print(f"→ Skipped {days_skipped} days with fewer than {context_size+1} valid tickers.")
        if drop_nans:
            print(f"→ Dropped {tickers_dropped} ticker‐positions due to NaNs across all days.")

    return out
