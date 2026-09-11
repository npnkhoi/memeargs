from extensions.base import AnnotationProcessor
from extensions.typing import Annotations
from extensions.typing.eval import Evaluation
from extensions.const import EVAL_ASPECTS

class EvaluationCollector(AnnotationProcessor):
    def process(self, annotations: Annotations, logger=None):
        """
        Based on the aspects in EVAL_ASPECTS, populate the evaluation object

        Expect exactly one of the following pairs of fields to be present:
        - {aspect}_evaluation
        - {aspect}_evaluation_by_gemini

        If both are present, use the gemini evaluation.
        If neither are present, raise an error.
        If only one is present, use that one.

        Return a dictionary of the evaluations.
        """
        # def get_camel_case(snake_case: str) -> str:
        #     tokens = snake_case.split("_")
        #     camel_case = "".join([tok.capitalize() for tok in toks])
        #     return camel_case
        
        all_evals = {}
        for aspect in EVAL_ASPECTS:
            qwen_source_name = aspect + "_evaluation"
            gemini_source_name = qwen_source_name + "_by_gemini"
            gemini_evaluation = getattr(annotations, gemini_source_name)
            qwen_evaluation = getattr(annotations, qwen_source_name)
            assert int(gemini_evaluation is not None) + int(qwen_evaluation is not None) <= 1, f"At most one of {qwen_source_name} or {gemini_source_name} should be present"
            evaluation = gemini_evaluation if gemini_evaluation is not None else qwen_evaluation
            if evaluation is None:
                evaluation = {
                    "score": None,
                    "reason": "Model failed to evaluate",
                }
            all_evals[aspect] = evaluation
        
        return Evaluation(**all_evals)