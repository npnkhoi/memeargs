import re
import signal
import threading


def get_word_count(text: str) -> int:
    from nltk.tokenize import word_tokenize

    return len(word_tokenize(text))


def get_lexical_diversity(text: str) -> float:
    from nltk.tokenize import word_tokenize
    from lexical_diversity.lex_div import mtld

    tokens = word_tokenize(text)
    if len(tokens) < 10:
        return 0.0
    return mtld(text)


def build_metric_suffix(essay_text: str, metrics: list) -> str:
    """Build constraint text for metric targets.

    Args:
        essay_text: Text to calculate metrics from
        metrics: List of ControlMetric enum values

    Returns:
        Formatted constraint text or empty string
    """
    from extensions.typing.config import ControlMetric

    lines = []
    for metric in metrics:
        if metric == ControlMetric.WORD_COUNT:
            wc = get_word_count(essay_text)
            lines.append(f"- Generate around {wc} words.")
        elif metric == ControlMetric.LEXICAL_DIVERSITY:
            ld = get_lexical_diversity(essay_text)
            lines.append(
                f"- Have a lexical diversity around {ld:.2f}. "
                "Lexical diversity measures how varied your vocabulary is. "
                "It is calculated via MTLD (Measure of Textual Lexical Diversity): "
                "the average length of sequential word strings that maintain a TTR "
                "(type-token ratio) above a threshold (0.72)."
            )
    if lines:
        return "\n\nConstraints:\n" + "\n".join(lines)
    return ""


# allows lockout of termination while writing to results file
class ShutdownFlag:
    def __init__(self):
        self.lock = threading.Lock()

        def raise_flag(_signum, _frame):
            with self.lock:
                raise InterruptedError()

        signal.signal(signal.SIGTERM, raise_flag)
        signal.signal(signal.SIGINT, raise_flag)


def camel_to_snake(camel_case_string: str) -> str:
    """
    Converts a CamelCase string to snake_case.

    Args:
        camel_case_string: The input string in CamelCase.

    Returns:
        The converted string in snake_case.

    # Example usage
    camel_string_1 = "myVariableName"
    snake_string_1 = camel_to_snake(camel_string_1)
    print(f"'{camel_string_1}' converted to snake_case: '{snake_string_1}'")

    camel_string_2 = "AnotherExampleString"
    snake_string_2 = camel_to_snake(camel_string_2)
    print(f"'{camel_string_2}' converted to snake_case: '{snake_string_2}'")

    camel_string_3 = "HTTPStatus"
    snake_string_3 = camel_to_snake(camel_string_3)
    print(f"'{camel_string_3}' converted to snake_case: '{snake_string_3}'")
    """
    # Use a regex to find uppercase letters that are not at the beginning
    # and insert an underscore before them.
    # Then, convert the entire string to lowercase.
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", camel_case_string)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    snake_case_string = s.lower()
    return snake_case_string
