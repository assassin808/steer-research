# [harness patch] lazy imports: toxicity needs `datasets`, dictatorgame needs statsmodels;
# neither is installed in our env and neither is needed for truthfulqa.
TASK_NAMES = ("truthfulqa", "twinviews", "toxicity", "dictatorgame")


def _cls(name):
    if name == "truthfulqa":
        from tasks.truthfulqa import TruthfulQA
        return TruthfulQA
    if name == "twinviews":
        from tasks.twinviews import TwinViews
        return TwinViews
    if name == "toxicity":
        from tasks.toxicity import Toxicity
        return Toxicity
    from tasks.dictatorgame import DictatorGame
    return DictatorGame


def get_task(name, model=None, character=None, persona_strength="none"):
    cls = _cls(name)
    if name == "dictatorgame":
        if model is None or character is None:
            raise ValueError("dictatorgame requires --model and --character")
        return cls(model, character, persona_strength)
    return cls()
