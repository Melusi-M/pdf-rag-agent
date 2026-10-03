import pytest

from pdf_rag_agent.arithmetic import evaluate_arithmetic


def test_evaluate_arithmetic_obeys_precedence() -> None:
    assert evaluate_arithmetic(
        "10 + 5 * 2"
    ) == 20


def test_evaluate_arithmetic_supports_parentheses() -> None:
    assert evaluate_arithmetic(
        "(10 + 5) * 2"
    ) == 30


def test_evaluate_arithmetic_supports_negative_values() -> None:
    assert evaluate_arithmetic(
        "-10 + 3"
    ) == -7


def test_evaluate_arithmetic_rejects_function_calls() -> None:
    with pytest.raises(
        ValueError,
        match="unsupported syntax",
    ):
        evaluate_arithmetic(
            "__import__('os').system('dir')"
        )


def test_evaluate_arithmetic_rejects_exponentiation() -> None:
    with pytest.raises(
        ValueError,
        match="operator is not supported",
    ):
        evaluate_arithmetic("10 ** 100")