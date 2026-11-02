import os
import json
import time
import math
from pathlib import Path

import pandas as pd
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "mohler.csv"
OUTPUT_CSV = BASE_DIR / "mohler_grok_predictions.csv"


XAI_API_KEY = "YOUR_API_KEY"

# Current Grok model.
MODEL = "grok-4.3"

# Start with 20 for a pilot.
# Change to None when you are ready for all 2442 responses.
LIMIT = None

# Number of retries if API call or JSON parsing fails.
MAX_RETRIES = 3

# Pause between requests.
SLEEP_BETWEEN_REQUESTS = 0


# ============================================================
# GROK / XAI CLIENT
# ============================================================


client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)


# ============================================================
# SCORE SCALE
# ============================================================

SCORE_VALUES = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
]


def score_key(score):
    score = float(score)

    if score.is_integer():
        return str(int(score))

    return str(score)


# ============================================================
# PROMPT
# ============================================================

def build_prompt(row):

    question = str(row["question"]).strip()
    reference_answer = str(
        row["reference_answer"]
    ).strip()

    student_answer = str(
        row["student_answer"]
    ).strip()

    allowed_scores = ", ".join(
        score_key(s)
        for s in SCORE_VALUES
    )

    prompt = f"""
You are an expert educational assessor grading a short-answer response.

Evaluate the student's answer using only the question and the reference answer.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

STUDENT ANSWER:
{student_answer}

The valid score scale is:

{allowed_scores}

A score of 0 represents a completely incorrect answer.
A score of 5 represents a fully correct answer.
Intermediate scores represent increasing levels of partial correctness.

Your task is NOT merely to choose one score.

Estimate the probability that each possible score is the appropriate human-assigned score.

Return ONLY a valid JSON object in exactly this structure:

{{
  "probabilities": {{
    "0": 0.01,
    "0.5": 0.01,
    "1": 0.02,
    "1.5": 0.03,
    "2": 0.05,
    "2.5": 0.08,
    "3": 0.10,
    "3.5": 0.15,
    "4": 0.20,
    "4.5": 0.20,
    "5": 0.15
  }}
}}

Requirements:

- Include all 11 score values.
- Do not include any additional scores.
- Every probability must be between 0 and 1.
- Probabilities must sum to 1.
- Assess semantic correctness rather than superficial word overlap.
- Do not provide explanations.
- Do not provide reasoning.
- Do not provide markdown.
- Return JSON only.
"""

    return prompt.strip()


# ============================================================
# JSON PARSING
# ============================================================

def parse_response(text):

    text = text.strip()

    # Remove accidental markdown fences.
    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "No JSON object found in response."
        )

    obj = json.loads(
        text[start:end + 1]
    )

    if "probabilities" not in obj:
        raise ValueError(
            "Response does not contain 'probabilities'."
        )

    raw_probs = obj["probabilities"]

    probabilities = {}

    for score in SCORE_VALUES:

        key = score_key(score)

        if key not in raw_probs:
            raise ValueError(
                f"Missing probability for score {key}"
            )

        value = float(
            raw_probs[key]
        )

        if (
            math.isnan(value)
            or math.isinf(value)
            or value < 0
        ):
            raise ValueError(
                f"Invalid probability: {value}"
            )

        probabilities[score] = value

    total = sum(
        probabilities.values()
    )

    if total <= 0:
        raise ValueError(
            "Probability sum is zero."
        )

    # Normalize small model rounding errors.
    probabilities = {
        score: value / total
        for score, value
        in probabilities.items()
    }

    return probabilities


# ============================================================
# DERIVED VALUES
# ============================================================

def calculate_prediction(probabilities):

    map_score = max(
        probabilities,
        key=probabilities.get
    )

    expected_score = sum(
        score * probability
        for score, probability
        in probabilities.items()
    )

    max_probability = max(
        probabilities.values()
    )

    entropy = 0.0

    for probability in probabilities.values():

        if probability > 0:

            entropy -= (
                probability
                * math.log(probability)
            )

    normalized_entropy = (
        entropy
        / math.log(len(SCORE_VALUES))
    )

    sorted_probs = sorted(
        probabilities.values(),
        reverse=True
    )

    margin = (
        sorted_probs[0]
        - sorted_probs[1]
    )

    margin_uncertainty = (
        1.0 - margin
    )

    max_probability_uncertainty = (
        1.0 - max_probability
    )

    variance = sum(
        probability
        * ((score - expected_score) ** 2)
        for score, probability
        in probabilities.items()
    )

    return {
        "predicted_score": map_score,
        "expected_score": expected_score,
        "max_probability": max_probability,
        "entropy": normalized_entropy,
        "max_uncertainty":
            max_probability_uncertainty,
        "margin_uncertainty":
            margin_uncertainty,
        "variance": variance,
    }


# ============================================================
# API CALL
# ============================================================

def grade_answer(row):

    prompt = build_prompt(
        row
    )

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    text = response.output_text

    probabilities = parse_response(
        text
    )

    derived = calculate_prediction(
        probabilities
    )

    return (
        probabilities,
        derived,
        text,
        response.id,
    )


# ============================================================
# EXISTING RESULTS
# ============================================================

def load_completed_ids():

    if not OUTPUT_CSV.exists():
        return set()

    try:

        existing = pd.read_csv(
            OUTPUT_CSV
        )

        if "response_id" not in existing.columns:
            return set()

        completed = existing[
            existing["status"] == "ok"
        ]

        return set(
            completed["response_id"]
            .astype(str)
        )

    except Exception:

        return set()


# ============================================================
# SAVE ONE RESULT
# ============================================================

def save_result(record):

    result = pd.DataFrame(
        [record]
    )

    result.to_csv(
        OUTPUT_CSV,
        mode="a",
        index=False,
        header=not OUTPUT_CSV.exists(),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Could not find:\n{INPUT_CSV}\n\n"
            "Put mohler.csv in the same folder "
            "as this script."
        )

    print(
        f"Reading dataset:\n{INPUT_CSV}"
    )

    df = pd.read_csv(
        INPUT_CSV
    )

    required_columns = [
        "item_id",
        "response_id",
        "question",
        "reference_answer",
        "student_answer",
        "human_score",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    if LIMIT is not None:

        df = df.head(
            LIMIT
        ).copy()

    completed_ids = (
        load_completed_ids()
    )

    total = len(df)

    print(
        f"\nProvider: xAI"
    )

    print(
        f"Model: {MODEL}"
    )

    print(
        f"Responses selected: {total}"
    )

    print(
        f"Already completed: "
        f"{len(completed_ids)}"
    )

    print(
        f"Output file:\n{OUTPUT_CSV}"
    )

    print(
        "\nStarting experiment...\n"
    )

    for number, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        response_id = str(
            row["response_id"]
        )

        if response_id in completed_ids:

            print(
                f"[{number}/{total}] "
                f"{response_id} "
                f"SKIPPED"
            )

            continue

        success = False
        last_error = ""

        for attempt in range(
            1,
            MAX_RETRIES + 1
        ):

            try:

                (
                    probabilities,
                    derived,
                    raw_response,
                    api_response_id,
                ) = grade_answer(
                    row
                )

                probability_json = json.dumps(
                    {
                        score_key(score):
                            probability
                        for score, probability
                        in probabilities.items()
                    },
                    ensure_ascii=False
                )

                record = {
                    "dataset":
                        row.get(
                            "dataset",
                            "Mohler"
                        ),

                    "item_id":
                        row["item_id"],

                    "response_id":
                        response_id,

                    "human_score":
                        float(
                            row["human_score"]
                        ),

                    "provider":
                        "xAI",

                    "model":
                        MODEL,

                    "predicted_score":
                        derived[
                            "predicted_score"
                        ],

                    "expected_score":
                        derived[
                            "expected_score"
                        ],

                    "absolute_error_map":
                        abs(
                            float(
                                derived[
                                    "predicted_score"
                                ]
                            )
                            - float(
                                row[
                                    "human_score"
                                ]
                            )
                        ),

                    "absolute_error_expected":
                        abs(
                            float(
                                derived[
                                    "expected_score"
                                ]
                            )
                            - float(
                                row[
                                    "human_score"
                                ]
                            )
                        ),

                    "probabilities":
                        probability_json,

                    "max_probability":
                        derived[
                            "max_probability"
                        ],

                    "entropy":
                        derived[
                            "entropy"
                        ],

                    "max_uncertainty":
                        derived[
                            "max_uncertainty"
                        ],

                    "margin_uncertainty":
                        derived[
                            "margin_uncertainty"
                        ],

                    "variance":
                        derived[
                            "variance"
                        ],

                    "api_response_id":
                        api_response_id,

                    "raw_response":
                        raw_response,

                    "status":
                        "ok",

                    "error":
                        "",
                }

                save_result(
                    record
                )

                success = True

                print(
                    f"[{number}/{total}] "
                    f"{response_id} "
                    f"human={row['human_score']} "
                    f"MAP="
                    f"{derived['predicted_score']} "
                    f"E="
                    f"{derived['expected_score']:.3f} "
                    f"U="
                    f"{derived['entropy']:.3f}"
                )

                break

            except Exception as exc:

                last_error = repr(
                    exc
                )

                print(
                    f"[{number}/{total}] "
                    f"{response_id} "
                    f"attempt {attempt} failed:"
                )

                print(
                    last_error
                )

                if attempt < MAX_RETRIES:

                    time.sleep(
                        2 ** attempt
                    )

        if not success:

            record = {
                "dataset":
                    row.get(
                        "dataset",
                        "Mohler"
                    ),

                "item_id":
                    row["item_id"],

                "response_id":
                    response_id,

                "human_score":
                    row["human_score"],

                "provider":
                    "xAI",

                "model":
                    MODEL,

                "predicted_score":
                    "",

                "expected_score":
                    "",

                "absolute_error_map":
                    "",

                "absolute_error_expected":
                    "",

                "probabilities":
                    "",

                "max_probability":
                    "",

                "entropy":
                    "",

                "max_uncertainty":
                    "",

                "margin_uncertainty":
                    "",

                "variance":
                    "",

                "api_response_id":
                    "",

                "raw_response":
                    "",

                "status":
                    "failed",

                "error":
                    last_error,
            }

            save_result(
                record
            )

        time.sleep(
            SLEEP_BETWEEN_REQUESTS
        )

    print(
        "\n============================"
    )

    print(
        "Experiment finished."
    )

    print(
        f"Results:\n{OUTPUT_CSV}"
    )

    print(
        "============================"
    )


if __name__ == "__main__":
    main()