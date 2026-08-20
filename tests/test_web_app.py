from fastapi.testclient import TestClient

from app import app


def test_home_page_uses_customer_facing_perkvector_branding():
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "Perk<span>Vector</span>" in response.text
    assert "Find the right credit card" in response.text
    assert "Gemini" not in response.text
    assert "25 verified cards" not in response.text


def test_results_show_calculation_and_source_provenance():
    response = TestClient(app).post(
        "/recommend",
        data={
            "dining": "500",
            "groceries": "400",
            "travel": "200",
            "gas": "100",
            "streaming": "50",
            "other": "300",
            "flights": "50",
            "hotels": "50",
            "transit": "50",
            "credit_score": "good",
            "max_annual_fee": "700",
            "preferred_rewards_type": "",
        },
    )

    assert response.status_code == 200
    assert "Year 1 Calculation" in response.text
    assert "Terms verified 2026-07-19" in response.text
    assert "View official issuer source" in response.text
    assert 'rel="noopener noreferrer"' in response.text
