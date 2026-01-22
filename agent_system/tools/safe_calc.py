from __future__ import annotations
import ast
import operator as op

_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,   # remove if you want to disallow exponent
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

class SafeMathError(ValueError):
    pass

def safe_eval_math(expr: str, *, max_len: int = 200) -> float:
    if not isinstance(expr, str) or not expr.strip():
        raise SafeMathError("Expression is empty.")
    if len(expr) > max_len:
        raise SafeMathError(f"Expression too long (>{max_len}).")

    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise SafeMathError(f"Invalid syntax: {e}") from e

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return float(n.value)
            raise SafeMathError("Only int/float constants are allowed.")

        if isinstance(n, ast.Num):  # pragma: no cover
            return float(n.n)

        if isinstance(n, ast.BinOp):
            op_type = type(n.op)
            if op_type not in _ALLOWED_BINOPS:
                raise SafeMathError(f"Operator not allowed: {op_type.__name__}")
            return float(_ALLOWED_BINOPS[op_type](_eval(n.left), _eval(n.right)))

        if isinstance(n, ast.UnaryOp):
            op_type = type(n.op)
            if op_type not in _ALLOWED_UNARYOPS:
                raise SafeMathError(f"Unary operator not allowed: {op_type.__name__}")
            return float(_ALLOWED_UNARYOPS[op_type](_eval(n.operand)))

        raise SafeMathError(f"Disallowed expression element: {type(n).__name__}")

    out = float(_eval(node))
    if out != out:
        raise SafeMathError("Result is NaN.")
    if out in (float("inf"), float("-inf")) or abs(out) > 1e308:
        raise SafeMathError("Result is infinite/too large.")
    return out
