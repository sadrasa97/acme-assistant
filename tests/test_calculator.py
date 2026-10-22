import math

import pytest

from app.agents.contracts import Status
from app.tools.calculator import CalculatorTool

calc = CalculatorTool()


@pytest.mark.parametrize("expr,expected", [
    ("1 + 2 * 3", 7), ("(10 - 4) / 3", 2), ("2 ** 10", 1024), ("-5 + 8", 3),
    ("percent_of(25, 200)", 12.5), ("avg(2, 4, 9)", 5), ("round(10 / 3, 2)", 3.33),
    ("max(1, 7, 3) - min(4, 2)", 5), ("percent_change(50, 75)", 50),
])
def test_valid_expressions(expr, expected):
    assert math.isclose(calc.evaluate(expr), expected, rel_tol=1e-9)


def test_variables_and_percentage():
    r = calc.run("part / total * 100", {"part": 371401.78, "total": 604712.61})
    assert r.status == Status.OK and math.isclose(r.result, 371401.78 / 604712.61 * 100)


@pytest.mark.parametrize("expr", ["1/0", "5 % 0", "percent_of(1, 0)", "0 ** -1", "10 // 0"])
def test_division_by_zero_is_controlled(expr):
    r = calc.run(expr)
    assert r.status == Status.ERROR and "zero" in r.error


@pytest.mark.parametrize("expr", [
    "__import__('os').system('id')", "open('/etc/passwd')", "().__class__", "x.y", "eval('1')",
    "[1,2][0]", "lambda: 1", "'a' * 3", "print(1)", "1 if 1 else 2", "exec('1')", "2 ** 9999",
    "10 ** 400", "", "1 +", "True + 1", "abs(1, 2, 3)", "sqrt(-1)",
])
def test_unsafe_or_invalid_rejected(expr):
    assert calc.run(expr).status == Status.ERROR


def test_unknown_variable_and_bad_inputs():
    assert calc.run("a + 1").status == Status.ERROR
    assert calc.run("a + 1", {"a": float("inf")}).status == Status.ERROR
    assert calc.run("a + 1", {"__x": 1}).status == Status.ERROR
