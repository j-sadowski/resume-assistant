
def calculate_overall_fit_and_tailoring_score(raw_fit_percentage: float, tailoring_level: str) -> float:
    """
    Calculates a composite score for overall fit and tailoring, capped at 100%.

    This function combines an objective assessment of a candidate's skills
    with a qualitative assessment of how well their application materials are
    customized for a specific role.

    Args:
        raw_fit_percentage (float): The candidate's inherent match to the
                                    job description's requirements (0-100%).
                                    This is a core, objective measure of skill alignment.
        tailoring_level (str): A qualitative assessment of how well the
                               application materials (resume, cover letter) are
                               customized. Options: "Exceptional", "Very Well",
                               "Well", "Moderate", "Generic".

    Returns:
        float: The combined overall fit and tailoring score (0-100%).
    """
    tailoring_boost_points = {
        "Exceptional": 10,
        "Very Well": 7,
        "Well": 4,
        "Moderate": 1,
        "Generic": -5
    }

    initial_score = raw_fit_percentage + \
        tailoring_boost_points.get(tailoring_level, 0)
    overall_score = max(0, min(initial_score, 100))

    return overall_score


def calculate_time_decay(days_since_posted: int) -> float:
    """
    Calculates a time decay factor based on how long a job has been posted.

    This function represents the decreasing likelihood of an interview over time
    as the hiring process progresses and the candidate pool grows.

    Args:
        days_since_posted (int): The number of days since the job was posted.

    Returns:
        float: A decay factor between 0.0 and 1.0.
    """
    if days_since_posted <= 14:  # First 2 weeks
        return 1.0
    elif days_since_posted <= 28:  # 2 to 4 weeks
        return 0.8
    elif days_since_posted <= 56:  # 4 to 8 weeks
        return 0.5
    elif days_since_posted <= 84:  # 8 to 12 weeks
        return 0.2
    else:
        return 0.1


def calculate_interview_chance(raw_fit_percentage: float, tailoring_level: str, days_since_posted=0) -> float:
    """
    Calculates the final estimated chance of getting an interview as a percentage.

    This function combines the overall fit and tailoring score with a time decay factor.

    Args:
        raw_fit_percentage (float): Inherent match to the job description (0-100%).
        tailoring_level (str): Qualitative assessment of application customization.
        days_since_posted (int): Number of days since the job was posted.

    Returns:
        float: The estimated probability of an interview as a percentage (0-100%).
    """
    overall_score = calculate_overall_fit_and_tailoring_score(
        raw_fit_percentage, tailoring_level)
    time_decay = calculate_time_decay(days_since_posted)
    interview_chance = (overall_score / 100.0) * time_decay * 100.0

    return round(interview_chance, 2)
