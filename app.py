"""
SecureScan AI demo application.

Run locally::

    python app.py                        # uses checkpoints/best_model.pt
    python app.py --checkpoint my.pt     # explicit checkpoint
    python app.py --smoke-test           # UI only, untrained weights

Then open http://localhost:7860

Note on ``--smoke-test``: the model weights are deliberately random. It exists
so the interface can be exercised without training a model first, and the UI
labels the output as untrained so it is never mistaken for a real prediction.

Checkpoints are validated by :mod:`src.models.loader` before use. One that does
not match the architecture is a hard error, because a partially applied
checkpoint leaves randomly-initialised layers that still answer confidently.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make ``src`` importable regardless of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from src.models.loader import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    CheckpointError,
    build_model,
    load_checkpoint,
)
from src.models.securescan_model import SecureScanModel  # noqa: E402

# Defaults, overridable from the command line.
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_DIR / "best_model.pt"
TOKENIZER_NAME = "microsoft/codebert-base"
MAX_LENGTH = 512
MAX_CHARS = 10_000
# Decision threshold on P(vulnerable). Exposed because the default 0.5 is not
# necessarily right for a detector whose classes are heavily imbalanced.
DEFAULT_THRESHOLD = 0.5

EXAMPLE_CODE = """char buffer[10];
strcpy(buffer, input);
return 0;"""


class Predictor:
    """Wraps the tokenizer and model behind a single ``predict`` call.

    Args:
        model: Trained (or, in smoke-test mode, untrained) classifier.
        tokenizer: Matching tokenizer.
        threshold: P(vulnerable) above which a snippet is flagged.
        trained: False when the weights are known to be random. Surfaced in the
            UI so an untrained model is never presented as a real result.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        threshold: float = DEFAULT_THRESHOLD,
        trained: bool = True,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.trained = trained

    def predict(self, code: str, threshold: float | None = None) -> dict:
        """Classify a snippet.

        Args:
            code: Source code to analyse.
            threshold: Overrides the instance threshold for this call only. The
                UI passes its slider value here rather than writing it onto the
                predictor, which is shared across concurrent requests.

        Returns:
            A JSON-serialisable result dict. ``error`` is non-None on failure.
        """
        if not code or not code.strip():
            return _error("Please provide some source code.")

        if len(code) > MAX_CHARS:
            return _error(
                f"Code is {len(code):,} characters; the limit is "
                f"{MAX_CHARS:,}. Analyse a single function instead."
            )

        active_threshold = self.threshold if threshold is None else float(threshold)

        try:
            inputs = self.tokenizer(
                code,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
            )
            with torch.no_grad():
                logits = self.model(inputs["input_ids"], inputs["attention_mask"])
                probs = torch.softmax(logits, dim=-1)[0]
            # Read inside the guard: a model with an unexpected number of
            # outputs should surface as an error, not as an IndexError trace.
            p_vulnerable = float(probs[1])
            p_safe = float(probs[0])
        except Exception as exc:  # surfaced to the UI, not a stack trace
            return _error(f"Prediction failed: {exc}")

        # Truncation is measured in tokens, not whitespace-separated words: a
        # single long token (a minified line, a long string literal) can exceed
        # the window with very few spaces. A sequence that fills the window is
        # reported as truncated, which errs towards warning the user.
        truncated = int(inputs["input_ids"].shape[-1]) >= MAX_LENGTH

        flagged = p_vulnerable >= active_threshold
        result = {
            "error": None,
            "verdict": "Vulnerable" if flagged else "Safe",
            "p_vulnerable": round(p_vulnerable, 4),
            "p_safe": round(p_safe, 4),
            "confidence": f"{max(p_vulnerable, 1 - p_vulnerable):.2%}",
            "threshold": active_threshold,
            "truncated": truncated,
        }
        if not self.trained:
            result["warning"] = (
                "UNTRAINED WEIGHTS — this prediction is meaningless. "
                "Rerun without --smoke-test using a trained checkpoint."
            )
        return result


def _error(message: str) -> dict:
    """Build a result dict carrying only an error message."""
    return {"error": message, "verdict": None, "confidence": None}


def build_predictor(
    checkpoint: str | os.PathLike[str] | None,
    *,
    smoke_test: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
) -> Predictor:
    """Construct a :class:`Predictor` from a checkpoint or random weights.

    Args:
        checkpoint: Checkpoint path. Ignored when ``smoke_test`` is True.
        smoke_test: Build an untrained model instead of loading weights.
        threshold: Decision threshold on P(vulnerable).

    Returns:
        A ready-to-use predictor.

    Raises:
        CheckpointError: If the checkpoint does not fit the architecture.
        FileNotFoundError: If the checkpoint cannot be located.
    """
    print(f"Loading tokenizer {TOKENIZER_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # ``from_scratch`` builds the encoder from its published config instead of
    # downloading ~500MB of pretrained weights that would be overwritten
    # anyway: ``load_checkpoint`` refuses anything below 100% coverage, so every
    # parameter — encoder included — is replaced by the checkpoint's. In
    # smoke-test mode the random weights are the point.
    model = build_model(SecureScanModel, from_scratch=True)

    if smoke_test:
        print("SMOKE TEST MODE — weights are random, predictions are meaningless.")
        return Predictor(model, tokenizer, threshold, trained=False)

    path = checkpoint or DEFAULT_CHECKPOINT
    print(f"Loading checkpoint {path} ...")
    load_checkpoint(model, path)  # raises if it does not fit
    print("Checkpoint loaded.")
    return Predictor(model, tokenizer, threshold, trained=True)


def build_interface(predictor: Predictor):
    """Assemble the Gradio interface.

    Args:
        predictor: Backend used by the Analyse button.

    Returns:
        An unlaunched ``gr.Blocks`` instance.
    """
    import gradio as gr

    banner = (
        "### :warning: Smoke-test mode\n"
        "This instance is running **untrained, randomly-initialised weights**. "
        "Predictions below are meaningless and are shown only to exercise the "
        "interface."
        if not predictor.trained
        else ""
    )

    with gr.Blocks(title="SecureScan AI") as demo:
        gr.Markdown(
            """
            # SecureScan AI
            *Source code vulnerability detection — CodeBERT + BiLSTM + MLP*

            Paste a C/C++ or Python function to classify it as **Safe** or
            **Vulnerable**.
            """
        )
        if banner:
            gr.Markdown(banner)

        with gr.Row():
            with gr.Column():
                code_input = gr.Textbox(
                    label="Source code",
                    placeholder="Paste a single function here...",
                    lines=12,
                )
                with gr.Row():
                    submit_btn = gr.Button("Analyse", variant="primary")
                    example_btn = gr.Button("Load sample")
                threshold_slider = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    value=predictor.threshold,
                    step=0.05,
                    label="Decision threshold on P(vulnerable)",
                    info="Lower flags more code as vulnerable. Default 0.5.",
                )
            with gr.Column():
                output = gr.JSON(label="Result")

        submit_btn.click(
            fn=lambda code, thr: _predict_with(predictor, code, thr),
            inputs=[code_input, threshold_slider],
            outputs=output,
        )
        # A no-input click handler. The previous version wired a string into a
        # gr.Number input, which raised on every click.
        example_btn.click(fn=lambda: EXAMPLE_CODE, inputs=None, outputs=code_input)

        gr.Markdown(
            """
            ---
            ### About
            - **Model**: CodeBERT (6 of 12 layers frozen) + 2-layer BiLSTM + MLP
            - **Scope**: binary classification only — this build does not
              predict a CWE category
            - **Limitations**: best on C/C++ and Python; may miss obfuscated or
              multi-function vulnerabilities; a single function is the intended
              unit of analysis
            - **Status**: see `docs/RESULTS.md` for the current, reproducible
              evaluation. Metrics previously shown here were not reproducible
              from this repository.
            """
        )

    return demo


def _predict_with(predictor: Predictor, code: str, threshold: float) -> dict:
    """Apply a UI-selected threshold, then delegate to the predictor.

    The threshold travels as an argument rather than being written onto the
    predictor, so two concurrent requests cannot overwrite each other's setting.
    """
    return predictor.predict(code, threshold=threshold)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SecureScan AI demo app.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=f"Path to a trained checkpoint (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the UI with untrained weights to verify the interface.",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link. Off by default.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)

    try:
        predictor = build_predictor(
            args.checkpoint,
            smoke_test=args.smoke_test,
            threshold=args.threshold,
        )
    except (CheckpointError, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        print(
            "No trained weights were found. Options:\n"
            "  1. Train one:      python -m src.training.train\n"
            "  2. Download a release checkpoint into checkpoints/\n"
            "  3. Verify the UI:  python app.py --smoke-test\n",
            file=sys.stderr,
        )
        return 1

    build_interface(predictor).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
