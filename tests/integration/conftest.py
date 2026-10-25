import pytest
from tests.integration.support import invoke


@pytest.fixture
def runs():
    records = []
    yield records
    for record in records:
        invoke("stop", record["id"], "--json")

