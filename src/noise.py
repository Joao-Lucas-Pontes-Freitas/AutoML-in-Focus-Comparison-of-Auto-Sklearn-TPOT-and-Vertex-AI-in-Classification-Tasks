import math

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype


# ============================================================
# 1) LABEL NOISE (stratified label flipping in TRAIN)
# ============================================================
def label_noise(
    y_train: pd.DataFrame, p: float = 0.10, random_state: int = 42
) -> pd.DataFrame:
    """
    Applies stratified label noise (flip) to the TRAIN set.
    - Selects ceil(p * n_per_class) instances per class.
    - Changes the label of these instances to another class chosen uniformly at random.
    - Prints audit metadata (number of flips per class).

    Parameters:
        y_train: DataFrame with a single column of labels.
        p: Fraction per class to be corrupted (default=0.10).
        random_state: Seed for reproducibility (default=42).

    Returns:
        y_train with labels flipped as specified.
    """
    if not isinstance(y_train, pd.DataFrame) or y_train.shape[1] != 1:
        raise ValueError("y_train must be a DataFrame with exactly 1 column.")

    rng = np.random.default_rng(random_state)
    col_name = y_train.columns[0]
    y_original = y_train[col_name]
    y_new = y_original.copy()

    classes = pd.unique(y_original)
    flip_counts = {}

    for cls in classes:
        idx = y_original.index[y_original == cls]
        n_flip = math.ceil(p * len(idx))
        if n_flip == 0:
            flip_counts[cls] = 0
            continue

        chosen_idx = rng.choice(idx.to_numpy(), size=n_flip, replace=False)
        flip_counts[cls] = len(chosen_idx)

        # For each chosen index, pick a new label != cls
        others = [c for c in classes if c != cls]
        for i in chosen_idx:
            y_new.at[i] = rng.choice(others)

    # Audit
    total_flips = sum(flip_counts.values())
    print("=== AUDIT: label_noise ===")
    print(f"p (fraction per class): {p}, random_state: {random_state}")
    print("Flips per class:")
    for cls in classes:
        print(f"  - {cls}: {flip_counts.get(cls, 0)}")
    print(f"Total labels changed: {total_flips}")

    # Return in the same format (DataFrame)
    return pd.DataFrame({col_name: y_new}, index=y_train.index)


# ============================================================
# 2) DATA NOISE (IMAGES) - SALT & PEPPER stratified
# ============================================================
def data_noise_images(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    dataset_name: str,
    p: float = 0.10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Applies SALT & PEPPER noise to image data (MNIST or DIGITS) in TRAIN.
    - Stratified by class: selects ceil(p * n_instances_per_class).
    - For each selected instance, chooses ceil(p * n_columns) pixels (columns) DIFFERENT per instance.
    - Sets half of these pixels to 0 (pepper) and half to max (salt).
      * MNIST: max = 255
      * DIGITS: max = 16
    - Returns modified X_train (same format) and prints metadata.

    Parameters:
        X_train: DataFrame (rows=instances, columns=pixels).
        y_train: DataFrame 1 column with labels (for stratification).
        dataset_name: 'mnist_784' or 'digits'.
        p: Fraction of instances per class and fraction of columns per instance (default=0.10).
        random_state: Seed for reproducibility (default=42).

    Returns:
        X_train with noise added.
    """
    if dataset_name.lower() not in {"mnist_784", "digits"}:
        raise ValueError("dataset_name must be 'mnist_784' or 'digits'.")

    if not isinstance(y_train, pd.DataFrame) or y_train.shape[1] != 1:
        raise ValueError("y_train must be a DataFrame with exactly 1 column.")

    rng = np.random.default_rng(random_state)
    X_new = X_train.copy()
    y_series = y_train.iloc[:, 0]

    n_cols = X_train.shape[1]
    n_noisy_cols = math.ceil(p * n_cols)

    # Max pixel value according to dataset
    max_value = 255 if dataset_name.lower() == "mnist_784" else 16
    min_value = 0

    count_per_class = {}
    classes = pd.unique(y_series)

    for cls in classes:
        idx = y_series.index[y_series == cls]
        n_inst = len(idx)
        n_noisy = math.ceil(p * n_inst)
        if n_noisy == 0:
            count_per_class[cls] = 0
            continue

        chosen_idx = rng.choice(idx.to_numpy(), size=n_noisy, replace=False)
        count_per_class[cls] = len(chosen_idx)

        for i in chosen_idx:
            # Pick columns (pixels) specific to this instance
            chosen_cols = rng.choice(
                X_new.columns.to_numpy(), size=n_noisy_cols, replace=False
            )

            # Half pepper (0), half salt (max). If odd, extra goes to salt.
            rng.shuffle(chosen_cols)
            half = len(chosen_cols) // 2
            col_pepper = chosen_cols[:half]
            col_salt = chosen_cols[half:]

            # Assignment
            X_new.loc[i, col_pepper] = min_value
            X_new.loc[i, col_salt] = max_value

    # Audit
    print("=== AUDIT: data_noise_images (salt & pepper) ===")
    print(f"Dataset: {dataset_name}, p: {p}, random_state: {random_state}")
    print(
        f"Total columns (pixels): {n_cols}, noisy columns per instance: {n_noisy_cols}"
    )
    print("Instances changed per class:")
    for cls in classes:
        print(f"  - {cls}: {count_per_class.get(cls, 0)}")
    print("Salt/pepper ratio per instance: 50/50 (extra to 'salt' if odd)")

    return X_new


# ============================================================
# 3) DATA NOISE (TABULAR) - numeric (gaussian) + categorical (swap)
#     - 10% instances per class
#     - 10% numeric columns per instance
#     - 10% categorical columns per instance
# ============================================================
def data_noise_tabular(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    numerical_columns: list,
    categorical_columns: list,
    p: float = 0.10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Applies noise to TABULAR data in TRAIN, stratified:
    - Selects ceil(p * n_instances_per_class) per class.
    - For EACH selected instance:
        * Randomly picks ceil(p * number of numeric columns) and applies additive Gaussian noise:
          - noise ~ N(0, std_col), using std calculated on the entire column before noise;
          - clips values to [min_col, max_col] (calculated before noise);
          - if column is integer, rounds and preserves original dtype.
        * Randomly picks ceil(p * number of categorical columns) and swaps the value for another valid one (≠ current value).

    Note:
        - Columns are picked per instance (not fixed globally).
        - NaN in categoricals remains unchanged.

    Parameters:
        X_train: tabular training data.
        y_train: DataFrame 1 column with labels (for stratification).
        numerical_columns: names of numeric columns.
        categorical_columns: names of categorical columns.
        p: fraction (default=0.10).
        random_state: seed (default=42).

    Returns:
        Modified X_train.
    """
    if not isinstance(y_train, pd.DataFrame) or y_train.shape[1] != 1:
        raise ValueError("y_train must be a DataFrame with exactly 1 column.")

    rng = np.random.default_rng(random_state)
    X_new = X_train.copy()
    y_series = y_train.iloc[:, 0]

    # Prepare statistics for numeric columns (before noise)
    # std can be 0 (constant column); in this case, do not alter this column when picked.
    means = {}
    stds = {}
    mins = {}
    maxs = {}
    for col in numerical_columns:
        col_series = X_new[col]
        means[col] = col_series.mean()
        stds[col] = col_series.std(ddof=0)  # ddof=0 (population) for stability
        mins[col] = col_series.min()
        maxs[col] = col_series.max()

    # Prepare domains for categoricals (possible values, excluding NaN)
    categorical_domains = {}
    for col in categorical_columns:
        values = pd.unique(X_new[col].dropna())
        categorical_domains[col] = list(values)

    # Number to pick per instance
    n_numeric_cols = (
        math.ceil(p * len(numerical_columns)) if len(numerical_columns) > 0 else 0
    )
    n_categorical_cols = (
        math.ceil(p * len(categorical_columns)) if len(categorical_columns) > 0 else 0
    )

    instance_counts = {}
    classes = pd.unique(y_series)

    for cls in classes:
        idx = y_series.index[y_series == cls]
        n_inst = len(idx)
        n_noisy = math.ceil(p * n_inst)
        instance_counts[cls] = n_noisy

        if n_noisy == 0:
            continue

        chosen_idx = rng.choice(idx.to_numpy(), size=n_noisy, replace=False)

        for i in chosen_idx:
            # --- NUMERIC: pick columns for this instance ---
            if n_numeric_cols > 0:
                chosen_num_cols = rng.choice(
                    numerical_columns,
                    size=min(n_numeric_cols, len(numerical_columns)),
                    replace=False,
                )
                for col in chosen_num_cols:
                    std_col = stds[col]
                    if std_col is None or np.isnan(std_col) or std_col == 0:
                        # Constant column (or invalid std): do not alter
                        continue

                    current_value = X_new.at[i, col]
                    noise = rng.normal(0.0, std_col)
                    new_value = current_value + noise

                    # Clip to [min, max] observed
                    new_value = max(mins[col], min(maxs[col], new_value))

                    # If column is integer, round and preserve dtype
                    if is_integer_dtype(X_new[col].dtype):
                        # handle nulls in integers (pandas 'Int64') — here we do not generate NaN
                        new_value = int(round(new_value))

                    X_new.at[i, col] = new_value

            # --- CATEGORICAL: pick columns for this instance ---
            if n_categorical_cols > 0:
                chosen_cat_cols = rng.choice(
                    categorical_columns,
                    size=min(n_categorical_cols, len(categorical_columns)),
                    replace=False,
                )
                for col in chosen_cat_cols:
                    current_value = X_new.at[i, col]
                    if pd.isna(current_value):
                        # Keep NaN
                        continue

                    domain = categorical_domains.get(col, [])
                    # If no alternative different, do not alter
                    alternatives = [v for v in domain if v != current_value]
                    if len(alternatives) == 0:
                        continue

                    X_new.at[i, col] = rng.choice(alternatives)

    # Audit
    print("=== AUDIT: data_noise_tabular ===")
    print(f"p: {p}, random_state: {random_state}")
    print(
        f"Num numeric columns: {len(numerical_columns)} | picked per instance: {n_numeric_cols}"
    )
    print(
        f"Num categorical columns: {len(categorical_columns)} | picked per instance: {n_categorical_cols}"
    )
    print("Instances changed per class:")
    for cls in classes:
        print(f"  - {cls}: {instance_counts.get(cls, 0)}")

    return X_new


# ============================================================
# 4) DATA NOISE (TEXT) - word removal
#     - 10% instances per class
#     - 20% chance of removal per word
# ============================================================
def data_noise_text(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    p: float = 0.10,
    random_state: int = 42,
    col_name: str = None,
) -> pd.DataFrame:
    """
    Applies textual noise by WORD REMOVAL in TRAIN, stratified:
    - Selects ceil(p * n_instances_per_class) per class.
    - For each selected text:
        * Tokenizes by space (simple split).
        * Removes each word with probability 0.20.
        * Ensures at least 1 word (if empty, keeps the first).
        * Tries to ensure the selected row is actually changed
            (if no word was removed and there are >= 2 tokens, removes 1 random token).
    - Preserves case and punctuation (just removes whole tokens).
    - Uses the text column passed as parameter.

    Parameters:
        X_train (pd.DataFrame): DataFrame with a text column.
        y_train (pd.DataFrame): DataFrame 1 column with labels (for stratification).
        p (float): fraction of instances per class to be affected (default=0.10).
        random_state (int): seed.

    Returns:
        pd.DataFrame: X_train with modified texts.
    """

    if not isinstance(y_train, pd.DataFrame) or y_train.shape[1] != 1:
        raise ValueError("y_train must be a DataFrame with exactly 1 column.")

    if col_name is None or col_name not in X_train.columns:
        raise ValueError("col_name must be the name of an existing column in X_train.")

    rng = np.random.default_rng(random_state)
    X_new = X_train.copy()
    y_series = y_train.iloc[:, 0]

    removal_prob = 0.20
    count_per_class = {}
    classes = pd.unique(y_series)

    for cls in classes:
        idx_cls = y_series.index[y_series == cls]
        n_inst = len(idx_cls)
        n_noisy = math.ceil(p * n_inst)
        count_per_class[cls] = n_noisy

        if n_noisy == 0:
            continue

        idx_instances = rng.choice(idx_cls.to_numpy(), size=n_noisy, replace=False)

        for i in idx_instances:
            text = X_new.at[i, col_name]

            # If not string (e.g., NaN), do not alter
            if not isinstance(text, str):
                continue

            tokens = text.split()
            if len(tokens) == 0:
                continue

            # Independent removal with probability 0.20 per token
            new_tokens = []
            for tk in tokens:
                if rng.random() < removal_prob:
                    continue
                new_tokens.append(tk)

            # >>> ENSURE CHANGE WHEN POSSIBLE:
            # If nothing was removed and there are at least 2 tokens, remove 1 random token
            if len(new_tokens) == len(tokens) and len(tokens) >= 2:
                idx_drop = rng.integers(0, len(tokens))
                new_tokens = tokens[:idx_drop] + tokens[idx_drop + 1 :]

            # Ensure at least 1 token
            if len(new_tokens) == 0:
                new_tokens = [tokens[0]]

            X_new.at[i, col_name] = " ".join(new_tokens)

    # Audit
    print("=== AUDIT: data_noise_text ===")
    print(
        f"p: {p}, random_state: {random_state}, removal_prob_per_word: {removal_prob}"
    )
    print("Instances changed per class:")
    for cls in classes:
        print(f"  - {cls}: {count_per_class.get(cls, 0)}")

    return X_new


import keyword
import re
import unicodedata


def normalize_cols(column_names):
    seen = {}  # base -> counter (for suffixes _1, _2, ...)
    output = []

    for name in column_names:
        s = str(name).strip().lower()

        # normalize accents and remove diacritics
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))

        # standardize curly quotes and replace any sequence not [a-z0-9] with "_"
        s = s.replace("'", "'")
        s = re.sub(r"[^a-z0-9]+", "_", s)

        # collapse multiple "_" and remove "_" at ends
        s = re.sub(r"_+", "_", s).strip("_")

        # empty -> "col"
        if not s:
            s = "col"

        # avoid starting with digit
        if s[0].isdigit():
            s = f"col_{s}"

        # avoid Python reserved words
        if keyword.iskeyword(s):
            s = f"{s}_"

        # ensure uniqueness (_1, _2, ...) only if already exists
        base = s
        k = seen.get(base, 0)
        if k > 0:
            s = f"{base}_{k}"
        seen[base] = k + 1

        output.append(s)

    return output
