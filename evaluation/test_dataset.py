"""
evaluation/test_dataset.py
--------------------------
Curated test dataset for RAGAS evaluation.

Each EvalSample contains a question, expected ground-truth answer, and the
query route it is expected to hit.  Ground truths are authored from the
placement document so they can be used for context_recall and
answer_correctness scoring.

Route values mirror placement/query_router.py constants:
  - "structured_query"  → eligibility, stats, hiring filters
  - "trend_query"       → year-over-year package growth
  - "conflict_check"    → CGPA / package discrepancies
  - "interview_text"    → interview rounds, tips (vector path)
  - "generic_vector"    → open-ended questions (vector path)
  - "out_of_corpus"     → questions outside the document scope
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class EvalSample:
    """One evaluation unit consumed by PipelineRunner."""

    question: str
    ground_truth: str
    route: str
    tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Structured-query samples  (eligibility / hiring / stats / package filters)
# ---------------------------------------------------------------------------
STRUCTURED_SAMPLES: List[EvalSample] = [
    EvalSample(
        question="What is the minimum CGPA requirement for Amazon?",
        ground_truth="Amazon requires a minimum CGPA of 6.4.",
        route="structured_query",
        tags=["eligibility", "cgpa", "amazon"],
    ),
    EvalSample(
        question="What is the minimum CGPA requirement for Google?",
        ground_truth="Google requires a minimum CGPA of 7.0.",
        route="structured_query",
        tags=["eligibility", "cgpa", "google"],
    ),
    EvalSample(
        question="Which company offers the highest package in the dataset?",
        ground_truth="Google offers the highest package at 42.0 LPA.",
        route="structured_query",
        tags=["eligibility", "package", "ranking"],
    ),
    EvalSample(
        question="Which companies allow students with backlogs?",
        ground_truth=(
            "TCS allows up to 2 backlogs, Amazon allows up to 1 backlog, "
            "and Infosys allows up to 1 backlog."
        ),
        route="structured_query",
        tags=["eligibility", "backlogs"],
    ),
    EvalSample(
        question="What is the bond duration for Amazon?",
        ground_truth="Amazon requires a bond of 2 years.",
        route="structured_query",
        tags=["eligibility", "bond", "amazon"],
    ),
    EvalSample(
        question="Which companies have no bond requirement?",
        ground_truth=(
            "TCS, Google, Infosys, and Microsoft have no bond requirement (0 years)."
        ),
        route="structured_query",
        tags=["eligibility", "bond"],
    ),
    EvalSample(
        question="What is the hiring distribution for TCS?",
        ground_truth=(
            "TCS hiring distribution: SDE 120, Analyst 80, Officer 30, "
            "Intern 200. Total hires: 430."
        ),
        route="structured_query",
        tags=["hiring", "distribution", "tcs"],
    ),
    EvalSample(
        question="Which company hires the most interns?",
        ground_truth="TCS hires the most interns with 200 intern positions.",
        route="structured_query",
        tags=["hiring", "intern", "ranking"],
    ),
    EvalSample(
        question="Which company hires the most SDEs?",
        ground_truth="TCS hires the most SDEs with 120 SDE positions.",
        route="structured_query",
        tags=["hiring", "sde", "ranking"],
    ),
    EvalSample(
        question="What is the package offered by Microsoft?",
        ground_truth="Microsoft offers a package of 35.0 LPA.",
        route="structured_query",
        tags=["eligibility", "package", "microsoft"],
    ),
    EvalSample(
        question="Which company has the best package-to-CGPA ratio?",
        ground_truth=(
            "Google has the best package-to-CGPA ratio at 6.00 LPA per CGPA point "
            "(42.0 LPA / 7.0 CGPA)."
        ),
        route="structured_query",
        tags=["stats", "ratio"],
    ),
    EvalSample(
        question="What is the tech focus for Amazon?",
        ground_truth="Amazon's tech focus is C++.",
        route="structured_query",
        tags=["eligibility", "tech_focus", "amazon"],
    ),
]

# ---------------------------------------------------------------------------
# Trend samples  (year-over-year package growth)
# ---------------------------------------------------------------------------
TREND_SAMPLES: List[EvalSample] = [
    EvalSample(
        question="Which company had the highest package growth from 2021 to 2024?",
        ground_truth=(
            "Google had the highest package growth from 2021 to 2024 with "
            "an absolute increase of 7.0 LPA (35.0 → 42.0 LPA)."
        ),
        route="trend_query",
        tags=["trend", "growth", "ranking"],
    ),
    EvalSample(
        question="What was TCS's package trend from 2021 to 2024?",
        ground_truth=(
            "TCS package trend: 2021: 3.2 LPA | 2022: 3.5 LPA | 2023: 3.8 LPA | "
            "2024: 4.1 LPA. Absolute growth: +0.9 LPA (trend: up)."
        ),
        route="trend_query",
        tags=["trend", "tcs"],
    ),
    EvalSample(
        question="How did Amazon's package grow between 2021 and 2024?",
        ground_truth=(
            "Amazon's package grew from 22.0 LPA in 2021 to 28.6 LPA in 2024, "
            "an absolute growth of 6.6 LPA."
        ),
        route="trend_query",
        tags=["trend", "amazon"],
    ),
    EvalSample(
        question="Which company showed a year-over-year increase in package every year?",
        ground_truth=(
            "TCS, Amazon, and Google all showed a consistent upward package trend "
            "from 2021 to 2024."
        ),
        route="trend_query",
        tags=["trend", "growth"],
    ),
]

# ---------------------------------------------------------------------------
# Conflict samples  (CGPA / package discrepancies)
# ---------------------------------------------------------------------------
CONFLICT_SAMPLES: List[EvalSample] = [
    EvalSample(
        question="Is Amazon's CGPA cutoff 6.4 or 7.0? There seems to be a discrepancy.",
        ground_truth=(
            "There are conflicting records for Amazon. The official criteria states "
            "6.4 CGPA while the portal record lists 7.0 CGPA. Please verify with "
            "the official placement cell."
        ),
        route="conflict_check",
        tags=["conflict", "amazon", "cgpa"],
    ),
    EvalSample(
        question="Which companies have conflicting CGPA records?",
        ground_truth=(
            "Amazon has conflicting CGPA records: the official criteria states 6.4 "
            "while the portal shows 7.0."
        ),
        route="conflict_check",
        tags=["conflict", "cgpa"],
    ),
    EvalSample(
        question="What is the conflict for Amazon in the placement dataset?",
        ground_truth=(
            "Amazon has a CGPA conflict: official CGPA is 6.4 but the portal lists "
            "7.0. The package data (28.6 LPA) is consistent across both records."
        ),
        route="conflict_check",
        tags=["conflict", "amazon"],
    ),
]

# ---------------------------------------------------------------------------
# Interview samples  (vector retrieval path)
# ---------------------------------------------------------------------------
INTERVIEW_SAMPLES: List[EvalSample] = [
    EvalSample(
        question="What is the technical focus of Google's interview Round 1?",
        ground_truth=(
            "Google's Round 1 is a technical round focusing on DSA and Python "
            "coding problems."
        ),
        route="interview_text",
        tags=["interview", "google", "round1"],
    ),
    EvalSample(
        question="What tips are given for Amazon's interview preparation?",
        ground_truth=(
            "Amazon's interview preparation tip is to practice DSA extensively, "
            "especially linked lists, trees, and dynamic programming, and to be "
            "familiar with LLD concepts."
        ),
        route="interview_text",
        tags=["interview", "amazon", "tips"],
    ),
    EvalSample(
        question="How many interview rounds does Microsoft have?",
        ground_truth=(
            "Microsoft typically has multiple rounds including a coding round, "
            "technical rounds covering DSA and OS/DBMS, and an HR round."
        ),
        route="interview_text",
        tags=["interview", "microsoft", "rounds"],
    ),
    EvalSample(
        question="What is the focus of TCS's interview process?",
        ground_truth=(
            "TCS's interview process focuses on DSA and SQL fundamentals, with "
            "a technical round and an HR round."
        ),
        route="interview_text",
        tags=["interview", "tcs"],
    ),
    EvalSample(
        question="What are the key preparation topics for Infosys interviews?",
        ground_truth=(
            "Key preparation topics for Infosys interviews include DSA and Python. "
            "Candidates should also practice aptitude and verbal sections."
        ),
        route="interview_text",
        tags=["interview", "infosys", "topics"],
    ),
    EvalSample(
        question="Does Google have an HR round in its interview process?",
        ground_truth=(
            "Yes, Google includes an HR/managerial round in addition to its "
            "technical coding rounds."
        ),
        route="interview_text",
        tags=["interview", "google", "hr"],
    ),
]

# ---------------------------------------------------------------------------
# Generic vector samples  (open-ended, no structured route)
# ---------------------------------------------------------------------------
GENERIC_SAMPLES: List[EvalSample] = [
    EvalSample(
        question="What are the key topics tested by Amazon during placement?",
        ground_truth=(
            "Amazon tests DSA, C++, and Low-Level Design (LLD) during its "
            "placement process."
        ),
        route="generic_vector",
        tags=["generic", "amazon", "topics"],
    ),
    EvalSample(
        question="What programming languages are commonly required across companies?",
        ground_truth=(
            "Common programming languages required across companies include Java "
            "(TCS), C++ (Amazon, Microsoft), and Python (Google, Infosys)."
        ),
        route="generic_vector",
        tags=["generic", "languages"],
    ),
    EvalSample(
        question="Which company has the strictest eligibility criteria overall?",
        ground_truth=(
            "Google and Microsoft have the strictest eligibility criteria, both "
            "requiring a minimum CGPA of 7.0 with zero backlogs allowed."
        ),
        route="generic_vector",
        tags=["generic", "eligibility"],
    ),
    EvalSample(
        question="What is the average package offered by TCS?",
        ground_truth="TCS offers a package of 4.1 LPA.",
        route="generic_vector",
        tags=["generic", "tcs", "package"],
    ),
    EvalSample(
        question="Which company is most suitable for a student with a CGPA of 6.5 and 1 backlog?",
        ground_truth=(
            "A student with CGPA 6.5 and 1 backlog is eligible for Amazon "
            "(min CGPA 6.4, max 1 backlog) and Infosys (min CGPA 6.5, max 1 backlog)."
        ),
        route="generic_vector",
        tags=["generic", "eligibility", "filter"],
    ),
]

# ---------------------------------------------------------------------------
# Out-of-corpus samples  (should be handled gracefully, not scored by RAGAS)
# ---------------------------------------------------------------------------
OUT_OF_CORPUS_SAMPLES: List[EvalSample] = [
    EvalSample(
        question="What is Infosys's current stock price?",
        ground_truth="This information is not available in the placement documents.",
        route="out_of_corpus",
        tags=["out_of_corpus", "stock"],
    ),
    EvalSample(
        question="What is the work from home policy at Google?",
        ground_truth="This information is not available in the placement documents.",
        route="out_of_corpus",
        tags=["out_of_corpus", "wfh"],
    ),
]

# ---------------------------------------------------------------------------
# Aggregated views
# ---------------------------------------------------------------------------

#: All samples that exercise the structured-reasoning path (no vector search).
STRUCTURED_PATH_SAMPLES: List[EvalSample] = (
    STRUCTURED_SAMPLES + TREND_SAMPLES + CONFLICT_SAMPLES
)

#: All samples that exercise the vector-retrieval path.
VECTOR_PATH_SAMPLES: List[EvalSample] = INTERVIEW_SAMPLES + GENERIC_SAMPLES

#: Complete evaluation set (excludes out-of-corpus to avoid RAGAS confusion).
ALL_EVAL_SAMPLES: List[EvalSample] = STRUCTURED_PATH_SAMPLES + VECTOR_PATH_SAMPLES

#: Lookup by route name.
SAMPLES_BY_ROUTE: dict = {
    "structured_query": STRUCTURED_SAMPLES,
    "trend_query": TREND_SAMPLES,
    "conflict_check": CONFLICT_SAMPLES,
    "interview_text": INTERVIEW_SAMPLES,
    "generic_vector": GENERIC_SAMPLES,
    "out_of_corpus": OUT_OF_CORPUS_SAMPLES,
}
