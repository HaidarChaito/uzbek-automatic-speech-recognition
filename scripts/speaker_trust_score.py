import numpy as np
from scipy import stats


def wilson_score(upvotes, downvotes, confidence=0.95):
    """
    Calculate Wilson score confidence interval lower bound.

    Returns: score between 0 and 1 (lower bound of confidence interval)
    """
    n = upvotes + downvotes

    if n == 0:
        return 0.5  # Neutral for no votes

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = upvotes / n

    numerator = (
        p_hat
        + z * z / (2 * n)
        - z * np.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n)
    )
    denominator = 1 + z * z / n

    lower_bound = round(numerator / denominator, ndigits=4)

    # Ensure score is in valid range [0, 1]
    return np.clip(lower_bound, 0.0, 1.0)


def trust_score(all_data_df):
    """
    Calculate speaker trust score using Wilson lower-bound
    """
    grouped = all_data_df.groupby("client_id")[["up_votes", "down_votes"]].sum()

    # Compute Wilson score for each speaker
    grouped["speaker_trust_score"] = grouped.apply(
        lambda row: wilson_score(row["up_votes"], row["down_votes"]), axis=1
    )

    # Map back to original dataframe
    all_data_df["speaker_trust_score"] = all_data_df["client_id"].map(
        grouped["speaker_trust_score"]
    )

    return all_data_df
