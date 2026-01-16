import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_distribution_graphs(df: pd.DataFrame, duration_key="duration"):
    """Plots audio duration and transcription word count distributions.

    Expected audio duration key is "duration" - duration in seconds
    """
    _, axes = plt.subplots(1, 2, figsize=(10, 5))

    total_duration = df[duration_key].sum()
    total_duration_label = (
        f"Total: {total_duration / 3600:,.1f}h"
        if total_duration / 3600 >= 1
        else f"Total: {total_duration / 60:,.0f}min"
    )

    axes[0].hist(df[duration_key], bins=30, edgecolor="black")
    axes[0].axvline(
        df[duration_key].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {df[duration_key].mean():.1f}s",
    )
    axes[0].axvline(
        5,
        color="blue",
        linestyle="",
        linewidth=2,
        label=total_duration_label,
    )
    axes[0].set_xlabel("Duration (seconds)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Audio Duration Distribution")
    axes[0].grid(
        True,
        axis="y",
        alpha=0.3,
        linewidth=0.7,
    )
    axes[0].legend()

    axes[1].hist(df["word_count"], bins=30, edgecolor="black", color="orange")
    axes[1].axvline(
        df["word_count"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {df["word_count"].mean():.1f}',
    )
    axes[1].axvline(
        5,
        color="blue",
        linestyle="",
        linewidth=2,
        label=f'Total: {df["word_count"].sum():,.0f}',
    )
    axes[1].set_xlabel("Word Count")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Transcription Word Count Distribution")
    axes[1].grid(
        True,
        axis="y",
        alpha=0.3,
        linewidth=0.7,
    )
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_dataset_type_gender_distribution(all_data_df: pd.DataFrame):
    # Copy and reclassify genders
    df = all_data_df[["type", "gender", "duration"]].copy()
    df["gender"] = df["gender"].where(
        df["gender"].isin(["female_feminine", "male_masculine"]), "not_specified"
    )

    dataset_types = df["type"].unique().tolist()

    # Group by type and gender, sum durations
    grouped = (
        df.groupby(["type", "gender"], dropna=False)["duration"]
        .sum()
        .unstack(fill_value=0)
    )

    # Convert duration to hours
    grouped = grouped / 3600

    # Sort types by total duration
    grouped["total_duration"] = grouped.sum(axis=1)
    grouped = grouped.sort_values("total_duration")

    # Compute absolute total and validated duration
    absolute_total = grouped["total_duration"].sum(axis=0)
    absolute_total_label = f"Total duration: {absolute_total:,.1f} h"

    validated_duration = grouped["total_duration"].loc[dataset_types].sum()
    validated_duration_label = f"Total validated duration: {validated_duration:,.1f} h"
    grouped = grouped.drop(columns="total_duration")

    # Compute total duration per gender for legend
    total_per_gender = grouped.sum(axis=0)

    # Plot stacked bar chart
    types = grouped.index.tolist()
    genders = grouped.columns.tolist()
    x = np.arange(len(types))

    fig, ax = plt.subplots(layout="constrained")

    bottom = np.zeros(len(types))  # starting bottom for stacking

    colors = {
        "female_feminine": "#FF69B4",
        "male_masculine": "#1E90FF",
        "not_specified": "#A9A9A9",
    }

    min_height = 10  # min hours to show labels

    for gender in genders:
        values = grouped[gender].values
        gender_label = f"{gender} (total: {total_per_gender[gender]:.1f} h)"
        rects = ax.bar(
            x, values, bottom=bottom, label=gender_label, color=colors.get(gender)
        )
        bottom += values  # update bottom for next stack
        for rect, val in zip(rects, values):
            if val >= min_height:
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_y() + rect.get_height() / 2,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    ax.set_ylabel("Duration (hours)")
    ax.set_title("Total Duration by Dataset Type and Gender")
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=45, ha="right")
    ax.axvline(
        4,
        linestyle="",
        label=absolute_total_label,
    )
    ax.axvline(
        4,
        linestyle="",
        label=validated_duration_label,
    )
    ax.grid(
        True,
        axis="y",
        alpha=0.3,
        linewidth=0.7,
    )
    ax.legend()

    plt.show()


def plot_accent_region_and_age_distribution(df: pd.DataFrame):
    # Prepare accent data
    accent_percentages = df["accent_region"].value_counts(normalize=True) * 100
    major_accent = accent_percentages[accent_percentages >= 5]
    minor_accent_sum = accent_percentages[accent_percentages < 5].sum()
    accent_data = (
        pd.concat([major_accent, pd.Series({"All Other": minor_accent_sum})])
        if minor_accent_sum > 0
        else major_accent
    )

    # Prepare age data
    age_percentages = df["age"].value_counts(normalize=True) * 100
    major_age = age_percentages[age_percentages >= 5]
    minor_age_sum = age_percentages[age_percentages < 5].sum()
    age_data = (
        pd.concat([major_age, pd.Series({"All Other": minor_age_sum})])
        if minor_age_sum > 0
        else major_age
    )

    # Create side-by-side pie charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Accent pie chart
    ax1.pie(accent_data, labels=accent_data.index, autopct="%1.1f%%", startangle=90)
    ax1.set_title("Speakers by Accent Region", fontsize=14, fontweight="bold")

    # Age pie chart
    ax2.pie(age_data, labels=age_data.index, autopct="%1.1f%%", startangle=90)
    ax2.set_title("Speakers by Age", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_gender_pie_chart(df: pd.DataFrame):
    df["gender"] = df["gender"].where(
        df["gender"].isin(["female_feminine", "male_masculine"]), "not_specified"
    )

    grouped = df.groupby(["gender"], dropna=False)["duration"].sum()
    # Convert duration to hours
    grouped = grouped / 3600

    colors = [
        "#FF69B4",
        "#1E90FF",
        "#A9A9A9",
    ]

    def func(pct, total_duration):
        duration = pct / 100 * total_duration
        return f"{pct:.1f}%\n({duration:.2f} h)"

    grouped.plot(
        kind="pie",
        ylabel="",
        autopct=lambda pct: func(pct, grouped.sum()),
        colors=colors,
        startangle=90,
        counterclock=False,
    )
    plt.title("Gender Proportion")
    plt.show()


def plot_demographic_statistics(df: pd.DataFrame):
    _, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Count recordings per speaker
    speaker_counts = df["client_id"].value_counts()

    bins = [0, 10, 50, 200, float("inf")]
    labels = ["1-10", "11-50", "51-200", "200+"]
    speakers_grouped = (
        pd.cut(speaker_counts, bins=bins, labels=labels).value_counts().sort_index()
    )

    speakers_grouped.plot(kind="barh", ax=axes[0])
    axes[0].axvline(
        5,
        linestyle="",
        linewidth=2,
        label=f"Total speakers: {len(speaker_counts):,.0f}",
    )
    axes[0].set_xlabel("Number of speakers")
    axes[0].set_ylabel("Number of recordings")
    axes[0].set_title("Recordings Per Speaker")
    axes[0].grid(
        True,
        axis="x",
        alpha=0.3,
        linewidth=0.7,
    )
    axes[0].legend()

    # Pie chart
    age_grouped = df["age"].value_counts().sort_index()

    # Combine very small slices (<1.2%) into "Other"
    total = age_grouped.sum()
    mask = (age_grouped / total) < 0.012
    if mask.any():
        other_sum = age_grouped[mask].sum()
        age_grouped = age_grouped[~mask]
        age_grouped["Other"] = other_sum

    colors = [
        (0, 0, 1, 0.7),
        (0, 1, 0, 0.7),
        (0.24, 0.79, 0.79, 0.7),
        (0.6, 0.6, 0.6, 0.7),
        (1, 0, 0, 0.7),
        (1, 1, 0, 0.7),
        (1, 0, 1, 0.7),
    ]

    # Function to show labels only for ≥1%
    def autopct_func(pct):
        return f"{pct:.1f}%" if pct >= 1 else ""

    age_grouped.plot(
        kind="pie",
        ax=axes[1],
        autopct=autopct_func,
        startangle=90,
        counterclock=False,
        ylabel="",
        colors=colors,
    )
    axes[1].set_title("Age Distribution of Speakers")

    plt.tight_layout()
    plt.show()


def plot_speaker_trust_score_distribution(df: pd.DataFrame, data_frame_title=""):
    trust_scores = df["speaker_trust_score"]

    plt.figure(figsize=(8, 5))
    plt.hist(trust_scores, bins=40, density=True)

    plt.axvline(
        trust_scores.mean(),
        linestyle="--",
        color="red",
        label=f"Mean: {trust_scores.mean():.2f}",
    )
    plt.axvline(
        0.8,
        linestyle="",
        label=f"Std: {trust_scores.std():.2f}",
    )
    plt.axvline(
        0.8,
        linestyle="",
        label=f"Max: {trust_scores.max():.2f}",
    )
    plt.axvline(
        0.8,
        linestyle="",
        label=f"Speakers: {len(df['client_id'].unique()):,}",
    )

    plt_title = "Distribution of Speaker Trust Score"
    if len(data_frame_title.strip()) > 0:
        plt_title = f"{plt_title} in {data_frame_title.title()}"
    plt.title(plt_title)

    plt.xlabel("Speaker Trust Score")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.show()
