import itertools

import pytest

from wheelta_robinhood_agent.domain.enums import ExecutionMode
from wheelta_robinhood_agent.domain.gating import effective_execution_mode

CASES = list(itertools.product(ExecutionMode, [True, False], ExecutionMode))


@pytest.mark.parametrize(("requested", "armed", "ceiling"), CASES)
def test_live_only_when_requested_armed_and_permitted(
    requested: ExecutionMode, armed: bool, ceiling: ExecutionMode
) -> None:
    result = effective_execution_mode(requested, armed=armed, ceiling=ceiling)
    expected_live = requested is ExecutionMode.LIVE and armed and ceiling is ExecutionMode.LIVE
    assert result is (ExecutionMode.LIVE if expected_live else ExecutionMode.OFF)
