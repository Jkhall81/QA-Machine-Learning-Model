from typing import Optional
from qa_grader.config import REQUIRED_PHRASES

# Full agent script questions
SCRIPT_QUESTIONS = {
    1: "This is {{fullname}} with National Assistance Center on a recorded line. Am I speaking with {{first_name}} {{last_name}}. The reason for my call is we’re helping Veterans and Homeowners MAXIMIZE their benefits through one of our Mortgage programs",
    2: "What type of loan are you in?",
    3: "Have you or your spouse served in the military?",
    4: "I’m showing that’s a single-family residence, is that correct?",
    5: "OK perfect, just need to verify: That you have made 12 payments or more on this current loan?",
    6: "Any late payments over 30 days in the last year?",
    7: "Any Bankruptcies or foreclosures in the last 2 years?",
    8: "What is your employment status?",
    9: "Around how much do you currently owe on the property.",
    10: "How much do you think your home is worth?",
    11: "How much debt are you dealing with not including your home loan?",
    12: "If eligible for an equity disbursement how much cash would you be interested in?",
    13: "And what do you think your current credit score is?",
    14: "OK it sounds like your credit could potentially hold you back from this program. We do have a debt relief program that can help your financial situation if you are eligible.",
    15: "Do you have $10,000 or more in credit card debt?",
    16: "Are you currently taking advantage of a debt relief program?",
    17: "Who are you working with?",
    18: "Are you interested in speaking with another Debt Relief company so you can compare their service and savings to what your currently getting?",
    19: "Are you currently making your minimum payments?",
    20: "Do you have an email address that you would like to provide for future reference? (required to ask)."
}


def check_rules(transcript: str) -> Optional[str]:
    """
    Apply rule-based checks to a transcript.
    Returns:
        - "fail" if transcript breaks hard rules
        - "warning" if transcript misses soft requirements
        - None if rules pass (let ML handle grading)
    """
    text = transcript.lower()

    # Rule 1: must say "on a recorded line"
    if REQUIRED_PHRASES["recorded_line"] not in text:
        return "fail"

    # Rule 2: must ask for email (Q20)
    if REQUIRED_PHRASES["email"] not in text:
        return "warning"

    # All required checks passed
    return None


def check_questions_present(transcript: str) -> dict:
    """
    Check which script questions appear in the transcript.
    Returns a dict {question_number: bool}
    """
    text = transcript.lower()
    results = {}
    for qnum, qtext in SCRIPT_QUESTIONS.items():
        snippet = qtext.lower().split("?")[0]  # take a recognizable part
        results[qnum] = snippet in text
    return results
