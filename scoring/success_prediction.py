def calculate_overall_fit_and_tailoring_score(raw_fit_percentage: float, tailoring_level: str) -> float:
    """
    Calculates a composite score for overall fit and tailoring, capped at 100%.

    Args:
        raw_fit_percentage (float): Your inherent match to the job description's
                                    requirements and preferred skills (0-100%).
                                    This is your objective assessment of your skills.
        tailoring_level (str): A qualitative assessment of how well your application
                               materials (resume, cover letter) are customized.
                               Options: "Exceptional", "Very Well", "Well", "Moderate", "Generic".

    Returns:
        float: The combined overall fit and tailoring score (0-100%).
    """

    # Define how much 'boost' or 'penalty' each tailoring level provides
    # These are illustrative values and can be adjusted based on desired impact.
    tailoring_boost_points = {
        "Exceptional": 10,   # Significant positive impact
        "Very Well": 7,      # Strong positive impact
        "Well": 4,           # Moderate positive impact
        "Moderate": 1,       # Slight positive impact
        "Generic": -5        # Potential penalty for lack of effort
    }

    if not (0 <= raw_fit_percentage <= 100):
        raise ValueError("raw_fit_percentage must be between 0 and 100.")
    if tailoring_level not in tailoring_boost_points:
        raise ValueError(f"Invalid tailoring_level: '{tailoring_level}'. Must be one of {list(tailoring_boost_points.keys())}")

    # Get the boost/penalty based on tailoring
    boost = tailoring_boost_points[tailoring_level]

    initial_score = raw_fit_percentage + boost

    overall_score = max(0, min(initial_score, 100))

    return overall_score


def calculate_time_decay_factor(date_posted):
    """"
    Time Decay (0.0 - 1.0):

    This factor accounts for how long the job has been posted, reflecting the decreasing likelihood of an interview over time as the hiring process progresses.

    1.0: Applied within the first 1-2 weeks of posting.

    0.8: Applied within 2-4 weeks.

    0.5: Applied within 4-8 weeks (1-2 months).

    0.2: Applied within 8-12 weeks (2-3 months).

    0.1 or lower: Applied after 12+ weeks (3+ months), unless you have a direct, strong referral.
    """
    return 1