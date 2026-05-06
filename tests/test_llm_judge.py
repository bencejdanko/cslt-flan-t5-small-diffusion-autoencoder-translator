import argparse
import json
import tempfile
import unittest
from pathlib import Path

from translation import llm_judge


class FakeBackend(llm_judge.JudgeBackend):
    score_calls = 0

    def __init__(self, model: str, base_url: str, timeout: int, num_ctx: int):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.num_ctx = num_ctx

    def health_check(self) -> None:
        return None

    def score(self, ref: str, pred: str) -> dict[str, object]:
        type(self).score_calls += 1
        return {"semantic": 2, "tone": 3, "fluency": 4, "rationale": "fake"}


def args_for(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values = {
        "predictions_file": str(tmp_path / "eval_results.json"),
        "output": str(tmp_path / "llm_judge_results.json"),
        "backend": "fake",
        "model": "fake-model",
        "ollama_url": "http://localhost:11434",
        "limit": 0,
        "resume": False,
        "overwrite": False,
        "preflight": False,
        "batch_size": 25,
        "timeout": 120,
        "num_ctx": 2048,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(json.dumps({"predictions": rows}))


class LlmJudgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_backend = llm_judge.BACKENDS.get("fake")
        llm_judge.BACKENDS["fake"] = FakeBackend
        FakeBackend.score_calls = 0

    def tearDown(self) -> None:
        if self.old_backend is None:
            llm_judge.BACKENDS.pop("fake", None)
        else:
            llm_judge.BACKENDS["fake"] = self.old_backend

    def test_parse_judge_json_variants(self) -> None:
        self.assertEqual(
            llm_judge._parse_judge_json(
                '{"semantic": 5, "tone": 4, "fluency": 3, "rationale": "ok"}'
            ),
            {"semantic": 5, "tone": 4, "fluency": 3, "rationale": "ok"},
        )
        self.assertEqual(
            llm_judge._parse_judge_json(
                'prefix {"semantic": 9, "tone": 0, "fluency": 3.4, "rationale": "x"} suffix'
            ),
            {
                "semantic": 5,
                "tone": 1,
                "fluency": 3,
                "rationale": "x",
                "_clamped": True,
            },
        )
        self.assertIsNone(llm_judge._parse_judge_json("not json"))
        self.assertIsNone(
            llm_judge._parse_judge_json(
                '{"semantic": "bad", "tone": 4, "fluency": 3, "rationale": "x"}'
            )
        )

    def test_compute_aggregates_excludes_error_rows(self) -> None:
        agg = llm_judge.compute_aggregates([
            {
                "idx": 0,
                "ref": "a",
                "pred": "b",
                "semantic": 1,
                "tone": 2,
                "fluency": 5,
                "total": 8,
                "len_bucket": "short",
                "rationale": "x",
            },
            {
                "idx": 1,
                "semantic": None,
                "tone": None,
                "fluency": None,
                "total": 0,
                "len_bucket": "short",
                "_error": True,
            },
        ])
        self.assertEqual(agg["overall"]["semantic"]["mean"], 1.0)
        self.assertEqual(agg["overall"]["semantic"]["n"], 1)
        self.assertEqual(agg["by_length_bucket"]["short"]["n"], 1)

    def test_resume_validation_rejects_signature_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            out = tmp / "out.json"
            expected = {
                "backend": "fake",
                "model": "m1",
                "predictions_file": str(tmp / "eval_results.json"),
                "predictions_sha256": "abc",
                "n_total": 2,
                "limit": 0,
                "num_ctx": 2048,
            }
            out.write_text(json.dumps({
                "meta": {"run_signature": expected},
                "scores": [{"idx": 0}, {"idx": 1}],
            }))
            self.assertEqual(len(llm_judge.load_existing(out, expected, 2)), 2)

            for key, value in [
                ("model", "m2"),
                ("predictions_sha256", "def"),
                ("limit", 1),
                ("n_total", 1),
            ]:
                changed = dict(expected)
                changed[key] = value
                with self.assertRaises(SystemExit):
                    llm_judge.load_existing(out, changed, changed["n_total"])

    def test_existing_output_requires_resume_or_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pred_path = tmp / "eval_results.json"
            write_predictions(pred_path, [{"ref": "a", "pred": "b"}])

            first = args_for(tmp)
            self.assertEqual(llm_judge.run(first), 0)
            self.assertEqual(FakeBackend.score_calls, 1)

            with self.assertRaises(SystemExit):
                llm_judge.run(args_for(tmp))

            self.assertEqual(llm_judge.run(args_for(tmp, resume=True)), 0)
            self.assertEqual(FakeBackend.score_calls, 1)

            self.assertEqual(llm_judge.run(args_for(tmp, overwrite=True)), 0)
            self.assertEqual(FakeBackend.score_calls, 2)

    def test_resume_starts_fresh_when_output_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pred_path = tmp / "eval_results.json"
            write_predictions(pred_path, [{"ref": "a", "pred": "b"}])

            self.assertEqual(llm_judge.run(args_for(tmp, resume=True)), 0)
            self.assertTrue((tmp / "llm_judge_results.json").exists())
            self.assertEqual(FakeBackend.score_calls, 1)

    def test_empty_inputs_are_scored_errors_without_backend_score(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pred_path = tmp / "eval_results.json"
            write_predictions(pred_path, [{"ref": "", "pred": "nonempty"}])
            self.assertEqual(llm_judge.run(args_for(tmp)), 0)
            self.assertEqual(FakeBackend.score_calls, 0)

            data = json.loads((tmp / "llm_judge_results.json").read_text())
            self.assertEqual(data["meta"]["n_errors"], 1)
            self.assertTrue(data["scores"][0]["_error"])
            self.assertEqual(data["scores"][0]["rationale"], "EMPTY_INPUT")


if __name__ == "__main__":
    unittest.main()
