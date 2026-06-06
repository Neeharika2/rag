import pytest
from placement.query_router import route_query, ROUTE_INTERVIEW
from placement.models import RoutedQuery
from generation.answerer import Answerer
from unittest.mock import MagicMock


def test_multi_company_routing() -> None:
    # Test that multiple companies are detected correctly
    r = route_query("tcs interview experience vs amazon interview experience")
    assert r.route == ROUTE_INTERVIEW
    assert "Amazon" in r.detected_companies
    assert "TCS" in r.detected_companies
    # Amazon has length 6, TCS has length 3, so in len desc order, it detects both.
    assert r.detected_company == "Amazon"  # First matched company by length desc


def test_single_company_routing() -> None:
    # Test that single company is still detected and detected_companies contains only one element
    r = route_query("tell me about Google interview process")
    assert r.route == ROUTE_INTERVIEW
    assert r.detected_companies == ["Google"]
    assert r.detected_company == "Google"


def test_merge_filters_multi_company() -> None:
    # Mock Retriever and Generator
    retriever_mock = MagicMock()
    generator_mock = MagicMock()
    answerer = Answerer(retriever=retriever_mock, generator=generator_mock)

    # Test merge filters with a single company
    routed_single = RoutedQuery(
        query="google interview",
        route=ROUTE_INTERVIEW,
        confidence=0.85,
        detected_company="Google",
        detected_companies=["Google"],
    )
    filters_single = answerer._merge_filters({"section": "interview"}, routed_single, None)
    assert filters_single == {"section": "interview", "company": "Google"}

    # Test merge filters with multiple companies
    routed_multi = RoutedQuery(
        query="tcs vs amazon",
        route=ROUTE_INTERVIEW,
        confidence=0.85,
        detected_company="Amazon",
        detected_companies=["Amazon", "TCS"],
    )
    filters_multi = answerer._merge_filters({"section": "interview"}, routed_multi, None)
    assert filters_multi == {"section": "interview", "company": ["Amazon", "TCS"]}


def test_no_section_data_message_multi() -> None:
    retriever_mock = MagicMock()
    generator_mock = MagicMock()
    answerer = Answerer(retriever=retriever_mock, generator=generator_mock)

    msg_single = answerer._no_section_data_message("interview", "Amazon")
    assert "for Amazon" in msg_single

    msg_multi = answerer._no_section_data_message("interview", ["Amazon", "TCS"])
    assert "for Amazon, TCS" in msg_multi
