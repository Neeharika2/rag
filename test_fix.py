"""
Offline smoke-test for the hiring distribution fix.
Tests routing + reasoner without calling Gemini API.
"""
import sys
sys.path.insert(0, r"c:\Users\neeha\Documents\rag")

from placement.query_router import route_query, ROUTE_HIRING, ROUTE_STRUCTURED

# ─── 1. Router tests ───────────────────────────────────────────────────────────
print("=" * 60)
print("ROUTER TESTS")
print("=" * 60)

routing_cases = [
    # (query, expected_route, label)
    ("google hiring distribution by role",         ROUTE_HIRING,     "original failing query"),
    ("how many analysts does Google hire",         ROUTE_HIRING,     "role count query"),
    ("role-wise breakdown for TCS",                ROUTE_HIRING,     "role-wise keyword"),
    ("distribution of hires in Microsoft",         ROUTE_HIRING,     "distribution keyword"),
    ("which company hires most SDEs",              ROUTE_HIRING,     "most SDEs"),
    ("total hires for Infosys",                    ROUTE_HIRING,     "total hires"),
    ("Google eligibility CGPA",                    ROUTE_STRUCTURED, "eligibility → NOT hiring"),
    ("what is Amazon's package",                   ROUTE_STRUCTURED, "package → NOT hiring"),
    ("Google CGPA and hiring distribution",        ROUTE_HIRING,     "mixed: distribution wins"),
]

all_pass = True
for query, expected, label in routing_cases:
    routed = route_query(query)
    status = "✅ PASS" if routed.route == expected else "❌ FAIL"
    if routed.route != expected:
        all_pass = False
    print(f"  {status}  [{label}]")
    print(f"         query   : {query!r}")
    print(f"         got     : {routed.route!r}  (expected {expected!r})")
    print(f"         metric  : {routed.detected_metric}  company: {routed.detected_company}")
    print()

# ─── 2. Reasoner tests ────────────────────────────────────────────────────────
print("=" * 60)
print("REASONER TESTS (using live DB dataset)")
print("=" * 60)

try:
    from settings import Settings
    from ingestion.metadata_store import MetadataStore
    from placement.reasoner import StructuredReasoner

    settings = Settings.from_env()
    store = MetadataStore(settings.metadata_db_url)
    store.init_db()
    dataset = store.get_latest_placement_dataset()

    if dataset is None:
        print("⚠️  No dataset in DB — skipping reasoner tests")
    else:
        reasoner = StructuredReasoner(dataset)

        reasoner_cases = [
            ("google hiring distribution by role",    "Google", ["sde", "analyst", "officer", "intern"]),
            ("how many analysts does Google hire",    "Google", ["analyst", "92"]),
            ("TCS hiring distribution by role",       "TCS",    ["sde", "analyst", "officer", "intern"]),
            ("Google eligibility CGPA",               "Google", ["7.4", "eligibility"]),
        ]

        for query, company, expected_keywords in reasoner_cases:
            result = reasoner.answer(query)
            answer_lower = result.answer.lower()
            kw_hits = [kw for kw in expected_keywords if kw.lower() in answer_lower]
            kw_miss = [kw for kw in expected_keywords if kw.lower() not in answer_lower]

            # Check evidence points to hiring record (has 'sde' key) for hiring queries
            evidence_ok = True
            if "distribution" in query or "analyst" in query.lower():
                evidence_ok = result.evidence and any("sde" in str(ev) for ev in result.evidence)

            status = "✅ PASS" if not kw_miss and evidence_ok else "❌ FAIL"
            if kw_miss or not evidence_ok:
                all_pass = False

            print(f"  {status}  [{query!r}]")
            print(f"         answer  : {result.answer}")
            print(f"         kw_hit  : {kw_hits}")
            if kw_miss:
                print(f"         kw_miss : {kw_miss}  ← MISSING IN ANSWER")
            if not evidence_ok:
                print(f"         evidence: {result.evidence}  ← no hiring record found")
            print()

except Exception as e:
    print(f"⚠️  Reasoner test failed with: {e}")
    import traceback; traceback.print_exc()

print("=" * 60)
print("ALL PASS" if all_pass else "SOME TESTS FAILED — see above")
print("=" * 60)
