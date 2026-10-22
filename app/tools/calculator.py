"""CalculatorTool: a whitelist-based AST evaluator. No eval/exec, no names except supplied variables,
no attribute access, no imports, no arbitrary calls."""
from __future__ import annotations

import ast
import math
import operator
import statistics
from typing import Callable

from app.agents.contracts import CalculatorResult, Status
from app.core.errors import ToolError

MAX_EXPR_LENGTH = 300
MAX_NODES = 100
MAX_ABS_VALUE = 1e15
MAX_EXPONENT = 100

_BINOPS: dict[type, Callable] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARYOPS: dict[type, Callable] = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _avg(*xs: float) -> float:
    if not xs:
        raise ToolError("avg() needs at least one value")
    return statistics.fmean(xs)


def _percent_of(part: float, whole: float) -> float:
    """part as a percentage of whole."""
    if whole == 0:
        raise ToolError("division by zero in percent_of(part, whole)")
    return part / whole * 100.0


def _percent_change(old: float, new: float) -> float:
    if old == 0:
        raise ToolError("division by zero in percent_change(old, new)")
    return (new - old) / abs(old) * 100.0


def _sqrt(x: float) -> float:
    if x < 0:
        raise ToolError("sqrt of a negative number")
    return math.sqrt(x)


_FUNCTIONS: dict[str, Callable] = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": lambda *xs: sum(xs),
    "avg": _avg, "mean": _avg, "sqrt": _sqrt, "floor": math.floor, "ceil": math.ceil,
    "percent_of": _percent_of, "percentage": _percent_of, "percent_change": _percent_change,
}
_CONSTANTS = {"pi": math.pi}


class CalculatorTool:
    name = "calculator"

    def evaluate(self, expression: str, variables: dict[str, float] | None = None) -> float:
        variables = variables or {}
        if not isinstance(expression, str) or not expression.strip():
            raise ToolError("empty expression")
        if len(expression) > MAX_EXPR_LENGTH:
            raise ToolError("expression too long")
        for k, v in variables.items():
            if not k.isidentifier() or k.startswith("_"):
                raise ToolError(f"invalid variable name: {k!r}")
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ToolError(f"variable {k!r} is not a finite number")
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError as exc:
            raise ToolError(f"invalid expression syntax: {exc.msg}") from exc
        if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
            raise ToolError("expression too complex")
        value = self._eval(tree.body, variables)
        return self._check(value)

    def run(self, expression: str, variables: dict[str, float] | None = None) -> CalculatorResult:
        """Never raises for expected failures; reports them in the result."""
        variables = dict(variables or {})
        try:
            value = self.evaluate(expression, variables)
            return CalculatorResult(status=Status.OK, expression=expression, inputs=variables, result=value)
        except ToolError as exc:
            return CalculatorResult(status=Status.ERROR, expression=expression, inputs=variables, error=exc.detail)

    # ---- internals ----
    @staticmethod
    def _check(value) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolError("expression did not evaluate to a number")
        if not math.isfinite(value) or abs(value) > MAX_ABS_VALUE:
            raise ToolError("result is not finite or out of range")
        return float(value)

    def _eval(self, node: ast.AST, variables: dict[str, float]):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ToolError("only numeric literals are allowed")
            return node.value
        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise ToolError(f"unknown name: {node.id}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = self._eval(node.left, variables), self._eval(node.right, variables)
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ToolError("division by zero")
            if isinstance(node.op, ast.Pow):
                if abs(right) > MAX_EXPONENT or abs(left) > MAX_ABS_VALUE:
                    raise ToolError("exponent too large")
                if left == 0 and right < 0:
                    raise ToolError("division by zero")
            try:
                result = _BINOPS[type(node.op)](left, right)
            except (OverflowError, ValueError, ZeroDivisionError) as exc:
                raise ToolError(f"arithmetic error: {exc}") from exc
            if isinstance(result, complex):
                raise ToolError("complex results are not supported")
            return self._check(result)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](self._eval(node.operand, variables))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
                raise ToolError("function not allowed")
            if node.keywords:
                raise ToolError("keyword arguments are not allowed")
            args = [self._eval(a, variables) for a in node.args]
            try:
                return self._check(_FUNCTIONS[node.func.id](*args))
            except ToolError:
                raise
            except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
                raise ToolError(f"invalid call to {node.func.id}(): {exc}") from exc
        raise ToolError(f"unsupported syntax: {type(node).__name__}")
