"""Local, deterministic GSM8K CoT and HellaSwag task adapters."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_GSM8K_PATH = Path("/share/project/mengyc/data/test.jsonl")
DEFAULT_HELLASWAG_PATH = Path("/share/project/mengyc/data/hellaswag_val.jsonl")

GSM8K_FEWSHOTS = (
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?", "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?", "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?", "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?", "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?", "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?", "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?", "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?", "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."),
)


@dataclass(frozen=True)
class BenchmarkExample:
    sample_id: str
    prompt: str
    target: Any
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class SampleScore:
    sample_id: str
    prediction: Any
    target: Any
    correct: bool


def _read_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    if not rows:
        raise ValueError(f"no samples loaded from {path}")
    return rows


def _preprocess_hellaswag(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def load_task_examples(
    task_name: str,
    limit: int | None,
    *,
    gsm8k_path: Path = DEFAULT_GSM8K_PATH,
    hellaswag_path: Path = DEFAULT_HELLASWAG_PATH,
) -> list[BenchmarkExample]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if task_name == "gsm8k_cot":
        prefix = "\n\n".join(f"Q: {question}\nA: {answer}" for question, answer in GSM8K_FEWSHOTS)
        return [
            BenchmarkExample(
                f"gsm8k_cot:{index}",
                f"{prefix}\n\nQ: {row['question']}\nA:",
                row["answer"].split("####")[-1].strip(),
            )
            for index, row in enumerate(_read_jsonl(gsm8k_path, limit))
        ]
    if task_name == "hellaswag":
        examples = []
        for index, row in enumerate(_read_jsonl(hellaswag_path, limit)):
            context = row["ctx_a"] + " " + row["ctx_b"].capitalize()
            examples.append(
                BenchmarkExample(
                    f"hellaswag:{index}",
                    _preprocess_hellaswag(row["activity_label"] + ": " + context),
                    int(row["label"]),
                    tuple(_preprocess_hellaswag(choice) for choice in row["endings"]),
                )
            )
        return examples
    raise ValueError(f"unsupported task: {task_name}")


def _normalize_number(value: str) -> str:
    return value.replace(",", "").replace("$", "").strip().rstrip(".")


def score_generation(example: BenchmarkExample, generated: str) -> SampleScore:
    strict = re.findall(r"The answer is (\-?[0-9\.\,]+)\.", generated)
    flexible = re.findall(r"(-?[$0-9.,]{2,}|-?[0-9]+)", generated)
    prediction = _normalize_number((strict or flexible or [""])[-1])
    target = _normalize_number(str(example.target))
    return SampleScore(example.sample_id, prediction, target, prediction == target)


def score_choices(
    example: BenchmarkExample,
    loglikelihoods: list[float],
    token_counts: list[int],
) -> SampleScore:
    if len(example.choices) != 4 or len(loglikelihoods) != 4 or len(token_counts) != 4:
        raise ValueError("HellaSwag scoring requires exactly four choices")
    normalized = [score / count if count > 0 else float("-inf") for score, count in zip(loglikelihoods, token_counts)]
    prediction = max(range(4), key=normalized.__getitem__)
    target = int(example.target)
    return SampleScore(example.sample_id, prediction, target, prediction == target)
