"""gsm8k: grade-school math word problems (single-turn, in-process verification).

Loads questions from GSM8K; the model reasons and gives a final number prefixed with
`#### `. The reward verifies equivalence with `vf.verify_boxed_math_answer` directly
in-process, so neither the rollout nor `validate` provisions a container or sandbox.
"""

import re
from typing import Literal

import verifiers.v1 as vf

SYSTEM = (
    "Solve the grade-school math problem. Reason step by step, then give the final "
    "answer as a single number on the last line, prefixed with '#### ' (e.g. '#### 42')."
)


def _score_gsm8k(answer: str, reply: str | None) -> float:
    text = reply or ""
    matches = re.findall(r"####\s*(.+)", text)
    prediction = matches[-1].strip() if matches else text
    return vf.verify_boxed_math_answer(f"\\boxed{{{prediction}}}", answer)


class GSM8KTask(vf.Task):
    answer: str
    """The ground-truth final answer (the value after GSM8K's `####`)."""


class GSM8KConfig(vf.TasksetConfig):
    split: Literal["train", "test"] = "test"


class GSM8KTaskset(vf.Taskset[GSM8KTask, GSM8KConfig]):
    def load_tasks(self) -> list[GSM8KTask]:
        from datasets import load_dataset

        rows = load_dataset("openai/gsm8k", "main", split=self.config.split)
        return [
            GSM8KTask(
                idx=i,
                prompt=f"{SYSTEM}\n\n{row['question']}",
                answer=row["answer"].split("####")[-1].strip(),
            )
            for i, row in enumerate(rows)
        ]

    @vf.reward(weight=1.0)
    async def correct(self, task: GSM8KTask, trace: vf.Trace) -> float:
        return _score_gsm8k(task.answer, trace.last_reply)

    async def validate(self, task: GSM8KTask, runtime: vf.Runtime) -> bool:
        """Valid iff the verifier accepts the ground-truth answer as a well-formed `#### N`
        prediction — catching rows the verifier can't parse or grade."""
        return _score_gsm8k(task.answer, f"#### {task.answer}") == 1.0
