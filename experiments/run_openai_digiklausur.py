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

INPUT_CSV = BASE_DIR / "digiklausur.csv"

OUTPUT_CSV = (
    BASE_DIR
    / "digiklausur_openai_predictions.csv"
)


OPENAI_API_KEY = "YOUR_API_KEY"

# Use exactly the same model as for Mohler.
MODEL = "gpt-6.1-terra"


# First test with 20.
# Change to None for the complete dataset.
LIMIT = None


MAX_RETRIES = 3

SLEEP_BETWEEN_REQUESTS = 0


# ============================================================
# OPENAI CLIENT
# ============================================================

if not OPENAI_API_KEY:

    raise ValueError(
        "OPENAI_API_KEY is missing.\n\n"
        "Set it in PowerShell with:\n"
        '$env:OPENAI_API_KEY="YOUR_KEY"'
    )


client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# DIGIKLAUSUR SCORE SCALE
# ============================================================

SCORE_VALUES = [
    0,
    1,
    2,
]


def score_key(score):

    return str(
        int(score)
    )


# ============================================================
# PROMPT
# ============================================================

def build_prompt(row):

    question = str(
        row["question"]
    ).strip()

    reference_answer = str(
        row["reference_answer"]
    ).strip()

    student_answer = str(
        row["student_answer"]
    ).strip()


    prompt = f"""
You are an expert educational assessor grading a short-answer response.

Evaluate the student's answer using only the question and the reference answer.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

STUDENT ANSWER:
{student_answer}

The grading scale contains exactly three possible scores:

0 = Incorrect
1 = Partially correct
2 = Correct

Your task is NOT merely to choose one score.

Estimate the probability that each of the three possible scores is the appropriate human-assigned score.

Return ONLY valid JSON in exactly this structure:

{{
  "probabilities": {{
    "0": 0.10,
    "1": 0.30,
    "2": 0.60
  }}
}}

Requirements:

- Include all three scores: 0, 1, and 2.
- Do not include any additional scores.
- Every probability must be between 0 and 1.
- Probabilities must sum to 1.
- Assess semantic correctness and completeness.
- Do not grade based only on lexical similarity.
- A score of 2 should indicate that the essential meaning required by the reference answer is correctly expressed.
- A score of 1 should indicate meaningful but incomplete or partially incorrect knowledge.
- A score of 0 should indicate an incorrect, irrelevant, or substantially inadequate answer.
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

    if text.startswith("```"):

        text = text.replace(
            "```json",
            ""
        )

        text = text.replace(
            "```",
            ""
        )

        text = text.strip()


    start = text.find("{")
    end = text.rfind("}")


    if start == -1 or end == -1:

        raise ValueError(
            "No JSON object found."
        )


    obj = json.loads(
        text[
            start:end + 1
        ]
    )


    if "probabilities" not in obj:

        raise ValueError(
            "Response does not contain "
            "'probabilities'."
        )


    raw_probs = (
        obj["probabilities"]
    )


    probabilities = {}


    for score in SCORE_VALUES:

        key = score_key(
            score
        )


        if key not in raw_probs:

            raise ValueError(
                f"Missing probability "
                f"for score {key}"
            )


        value = float(
            raw_probs[key]
        )


        if (
            math.isnan(value)
            or math.isinf(value)
            or value < 0
            or value > 1
        ):

            raise ValueError(
                f"Invalid probability: "
                f"{value}"
            )


        probabilities[
            score
        ] = value


    total = sum(
        probabilities.values()
    )


    if total <= 0:

        raise ValueError(
            "Probability sum is zero."
        )


    # Normalize minor rounding differences.
    probabilities = {

        score:
            value / total

        for score, value
        in probabilities.items()
    }


    return probabilities


# ============================================================
# DERIVED VALUES
# ============================================================

def calculate_prediction(
    probabilities
):


    # MAP prediction
    map_score = max(
        probabilities,
        key=probabilities.get
    )


    # Expected score
    expected_score = sum(

        score * probability

        for score, probability
        in probabilities.items()
    )


    # Maximum predicted probability
    max_probability = max(
        probabilities.values()
    )


    # --------------------------------------------
    # Normalized entropy
    # --------------------------------------------

    entropy = 0.0


    for probability in (
        probabilities.values()
    ):

        if probability > 0:

            entropy -= (
                probability
                * math.log(
                    probability
                )
            )


    normalized_entropy = (
        entropy
        / math.log(
            len(SCORE_VALUES)
        )
    )


    # --------------------------------------------
    # Maximum-probability uncertainty
    # --------------------------------------------

    max_uncertainty = (
        1.0
        - max_probability
    )


    # --------------------------------------------
    # Margin uncertainty
    # --------------------------------------------

    sorted_probs = sorted(
        probabilities.values(),
        reverse=True
    )


    margin = (
        sorted_probs[0]
        - sorted_probs[1]
    )


    margin_uncertainty = (
        1.0
        - margin
    )


    # --------------------------------------------
    # Ordinal score variance
    # --------------------------------------------

    variance = sum(

        probability
        * (
            (
                score
                - expected_score
            ) ** 2
        )

        for score, probability
        in probabilities.items()
    )


    return {

        "predicted_score":
            map_score,

        "expected_score":
            expected_score,

        "max_probability":
            max_probability,

        "entropy":
            normalized_entropy,

        "max_uncertainty":
            max_uncertainty,

        "margin_uncertainty":
            margin_uncertainty,

        "variance":
            variance,
    }


# ============================================================
# API CALL
# ============================================================

def grade_answer(row):

    prompt = build_prompt(
        row
    )


    response = (
        client.responses.create(

            model=MODEL,

            input=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
    )


    text = (
        response.output_text
    )


    probabilities = (
        parse_response(
            text
        )
    )


    derived = (
        calculate_prediction(
            probabilities
        )
    )


    return (
        probabilities,
        derived,
        text,
        response.id,
    )


# ============================================================
# LOAD COMPLETED RESPONSES
# ============================================================

def load_completed_ids():

    if not OUTPUT_CSV.exists():

        return set()


    try:

        existing = pd.read_csv(
            OUTPUT_CSV
        )


        if (
            "response_id"
            not in existing.columns
        ):

            return set()


        completed = existing[
            existing["status"]
            == "ok"
        ]


        return set(
            completed[
                "response_id"
            ].astype(str)
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

        header=(
            not OUTPUT_CSV.exists()
        ),

        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():


    # --------------------------------------------
    # Read input
    # --------------------------------------------

    if not INPUT_CSV.exists():

        raise FileNotFoundError(

            f"Could not find:\n"
            f"{INPUT_CSV}\n\n"

            "Put digiklausur.csv "
            "in the same folder "
            "as this script."
        )


    print(
        f"Reading dataset:\n"
        f"{INPUT_CSV}"
    )


    df = pd.read_csv(
        INPUT_CSV
    )


    # --------------------------------------------
    # Required columns
    # --------------------------------------------

    required_columns = [

        "item_id",
        "response_id",
        "question",
        "reference_answer",
        "student_answer",
        "human_score",
    ]


    missing_columns = [

        column

        for column
        in required_columns

        if column
        not in df.columns
    ]


    if missing_columns:

        raise ValueError(

            "Missing columns: "
            + ", ".join(
                missing_columns
            )
        )


    # --------------------------------------------
    # Validate human labels
    # --------------------------------------------

    human_values = set(
        pd.to_numeric(
            df["human_score"],
            errors="coerce"
        )
        .dropna()
        .astype(int)
        .unique()
    )


    unexpected = (
        human_values
        - set(SCORE_VALUES)
    )


    if unexpected:

        raise ValueError(

            "Unexpected DigiKlausur "
            "human scores found: "
            f"{sorted(unexpected)}"
        )


    # --------------------------------------------
    # Pilot limit
    # --------------------------------------------

    if LIMIT is not None:

        df = df.head(
            LIMIT
        ).copy()


    completed_ids = (
        load_completed_ids()
    )


    total = len(df)


    print(
        f"\nDataset: DigiKlausur"
    )

    print(
        f"Model: {MODEL}"
    )

    print(
        f"Score scale: "
        f"{SCORE_VALUES}"
    )

    print(
        f"Responses selected: "
        f"{total}"
    )

    print(
        f"Already completed: "
        f"{len(completed_ids)}"
    )

    print(
        f"Output file:\n"
        f"{OUTPUT_CSV}"
    )

    print(
        "\nStarting experiment...\n"
    )


    # ========================================================
    # PROCESS RESPONSES
    # ========================================================

    for number, (_, row) in enumerate(

        df.iterrows(),

        start=1
    ):


        response_id = str(
            row["response_id"]
        )


        # ----------------------------------------
        # Resume support
        # ----------------------------------------

        if (
            response_id
            in completed_ids
        ):

            print(

                f"[{number}/{total}] "
                f"{response_id} "
                f"SKIPPED"
            )

            continue


        success = False

        last_error = ""


        # ----------------------------------------
        # Retry loop
        # ----------------------------------------

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


                probability_json = (
                    json.dumps(

                        {

                            score_key(score):
                                probability

                            for score, probability
                            in probabilities.items()
                        },

                        ensure_ascii=False
                    )
                )


                human_score = int(
                    row["human_score"]
                )


                # ====================================
                # RESULT RECORD
                # ====================================

                record = {

                    "dataset":
                        row.get(
                            "dataset",
                            "DigiKlausur"
                        ),

                    "item_id":
                        row["item_id"],

                    "response_id":
                        response_id,

                    "human_score":
                        human_score,

                    "model":
                        MODEL,


                    # ----------------------------
                    # Predictions
                    # ----------------------------

                    "predicted_score":
                        derived[
                            "predicted_score"
                        ],

                    "expected_score":
                        derived[
                            "expected_score"
                        ],


                    # ----------------------------
                    # Errors
                    # ----------------------------

                    "absolute_error_map":
                        abs(

                            float(
                                derived[
                                    "predicted_score"
                                ]
                            )

                            - human_score
                        ),

                    "absolute_error_expected":
                        abs(

                            float(
                                derived[
                                    "expected_score"
                                ]
                            )

                            - human_score
                        ),


                    # ----------------------------
                    # Probability distribution
                    # ----------------------------

                    "probabilities":
                        probability_json,


                    # ----------------------------
                    # Uncertainty
                    # ----------------------------

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


                    # ----------------------------
                    # API metadata
                    # ----------------------------

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

                    f"human="
                    f"{human_score} "

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

                    f"attempt "
                    f"{attempt} failed:"
                )


                print(
                    last_error
                )


                if (
                    attempt
                    < MAX_RETRIES
                ):

                    time.sleep(
                        2 ** attempt
                    )


        # ============================================
        # FAILED RECORD
        # ============================================

        if not success:


            record = {

                "dataset":
                    row.get(
                        "dataset",
                        "DigiKlausur"
                    ),

                "item_id":
                    row["item_id"],

                "response_id":
                    response_id,

                "human_score":
                    row["human_score"],

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


    # ========================================================
    # DONE
    # ========================================================

    print(
        "\n============================"
    )

    print(
        "DigiKlausur experiment "
        "finished."
    )

    print(
        f"Results:\n"
        f"{OUTPUT_CSV}"
    )

    print(
        "============================"
    )


if __name__ == "__main__":
    main()