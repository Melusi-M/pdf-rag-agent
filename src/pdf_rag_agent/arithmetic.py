import ast
import operator
from collections.abc import Callable


MAX_EXPRESSION_LENGTH = 200
MAX_ABSOLUTE_VALUE = 1_000_000_000_000
MAX_EXPRESSION_DEPTH = 20

BINARY_OPERATORS: dict[
    type[ast.operator],
    Callable[[float, float], float],
] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}

UNARY_OPERATORS: dict[
    type[ast.unaryop],
    Callable[[float], float],
] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate_arithmetic(
    expression: str,
) -> int | float:
    normalized_expression = expression.strip()

    if not normalized_expression:
        raise ValueError(
            "Arithmetic expression cannot be empty"
        )

    if len(normalized_expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(
            "Arithmetic expression is too long"
        )

    try:
        syntax_tree = ast.parse(
            normalized_expression,
            mode="eval",
        )
    except SyntaxError as error:
        raise ValueError(
            "Invalid arithmetic expression"
        ) from error

    return _evaluate_node(
        syntax_tree.body,
        depth=0,
    )


def _evaluate_node(
    node: ast.AST,
    depth: int,
) -> int | float:
    if depth > MAX_EXPRESSION_DEPTH:
        raise ValueError(
            "Arithmetic expression is too complex"
        )

    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return _validate_result(node.value)

    if isinstance(node, ast.BinOp):
        operation = BINARY_OPERATORS.get(
            type(node.op)
        )

        if operation is None:
            raise ValueError(
                "Arithmetic operator is not supported"
            )

        left = _evaluate_node(
            node.left,
            depth + 1,
        )
        right = _evaluate_node(
            node.right,
            depth + 1,
        )

        return _validate_result(
            operation(left, right)
        )

    if isinstance(node, ast.UnaryOp):
        operation = UNARY_OPERATORS.get(
            type(node.op)
        )

        if operation is None:
            raise ValueError(
                "Unary operator is not supported"
            )

        operand = _evaluate_node(
            node.operand,
            depth + 1,
        )

        return _validate_result(
            operation(operand)
        )

    raise ValueError(
        "Expression contains unsupported syntax"
    )


def _validate_result(
    value: int | float,
) -> int | float:
    if abs(value) > MAX_ABSOLUTE_VALUE:
        raise ValueError(
            "Arithmetic result exceeds the allowed limit"
        )

    return value