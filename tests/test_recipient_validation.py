from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import main
from app.models.recipient import Recipient, RecipientCreate

client = TestClient(main.app)

BAD = ["not-an-email", "founder@", "@startup.io", "a b@startup.io", "founder@startup", "", "founder@@startup.io"]


@pytest.mark.parametrize("address", BAD)
def test_malformed_recipient_addresses_are_rejected(address):
    with pytest.raises(ValidationError):
        RecipientCreate(email=address)


@pytest.mark.parametrize("address", ["founder@startup.io", "first.last+outreach@sub.startup.co.uk", "A.Founder@Startup.IO"])
def test_valid_recipient_addresses_are_accepted(address):
    assert RecipientCreate(email=address).email


def test_the_recipients_endpoint_answers_422_for_a_malformed_address():
    with patch.object(main.db, "insert") as insert:
        response = client.post("/recipients", json={"name": "Sam", "email": "founder@"})
    assert response.status_code == 422 and "email" in response.text
    insert.assert_not_called()                                                     # nothing bad is stored


def test_the_recipients_endpoint_still_saves_a_valid_one():
    with patch.object(main.db, "insert", side_effect=lambda table, data: {"id": "r1", **data}):
        response = client.post("/recipients", json={"name": "Sam", "email": "sam@startup.io", "role": "CTO"})
    assert response.status_code == 201 and response.json()["email"] == "sam@startup.io"


def test_outreach_for_a_recipient_stored_with_a_bad_address_is_stopped_before_any_email_is_created():
    rows = {
        "profiles": {"id": "p1", "name": "A", "email": "a@example.com"},
        "companies": {"id": "c1", "name": "Acme"},
        "recipients": {"id": "r1", "name": "Sam", "email": "not-an-email"},       # saved before validation existed
    }
    with patch.object(main.db, "fetch", side_effect=lambda table, id_: rows[table]), patch.object(main.db, "insert") as insert:
        response = client.post("/outreach", json={"profile_id": "p1", "company_id": "c1", "recipient_id": "r1"})
    assert response.status_code == 422 and "invalid email address" in response.json()["detail"]
    insert.assert_not_called()
