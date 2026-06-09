import sys
sys.path.insert(0, r"c:\Users\neeha\Documents\rag")

from settings import Settings
from ingestion.metadata_store import MetadataStore
from placement.reasoner import StructuredReasoner

settings = Settings.from_env()
store = MetadataStore(settings.metadata_db_url)
store.init_db()
dataset = store.get_latest_placement_dataset()

reasoner = StructuredReasoner(dataset)

cases = [
    ("google hiring distribution by role",  ["SDE", "Analyst", "Officer", "Intern", "198"], "hiring"),
    ("how many analysts does Google hire",  ["92"], "hiring"),
    ("TCS hiring distribution by role",     ["SDE", "Analyst", "Officer", "Intern", "244"], "hiring"),
    ("Google eligibility CGPA",             ["7.4", "eligibility"], None),
    ("what is Google package",              ["42.0", "LPA"], None),
]

all_pass = True
for query, kws, ev_key in cases:
    result = reasoner.answer(query)
    answer_l = result.answer.lower()
    missing = [k for k in kws if k.lower() not in answer_l]

    ev_ok = True
    if ev_key == "hiring":
        ev_ok = bool(result.evidence) and any("sde" in str(e) for e in result.evidence)

    ok = not missing and ev_ok
    if not ok:
        all_pass = False
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {query}")
    print(f"       answer: {result.answer}")
    print(f"       ev_ok : {ev_ok}")
    if missing:
        print(f"       MISS  : {missing}")
    print()

print("ALL PASS" if all_pass else "SOME TESTS FAILED")
