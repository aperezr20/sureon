import re
import os

SEQSUM_JUDGE_PROMPT = """\
You are grading a model's summary of a (surgical/procedural) video clip against a
reference summary. You see only the two texts, NOT the video, so you cannot verify
any claim that goes beyond the reference.

Question:
{question}

Ground Truth Answer:
{ground_truth}

Predicted Answer:
{predicted}

Your task:
1. Decide if the predicted answer is CORRECT or INCORRECT relative to the ground truth.

   Judge on two things, in this priority order:

   a) SEQUENCE & COVERAGE (primary): Do the predicted steps/structures/actions match
      the reference in content and order?
      - Be semantically flexible: accept paraphrases, reordered lists where order is
        not meaningful, and minor wording differences if the meaning matches.
      - Mark INCORRECT if any key step, structure, or item from the reference is
        missing, or if the order is wrong when order matters.

   b) FAITHFULNESS (secondary): Does the prediction stay consistent with the reference?
      - Detail present in the prediction but ABSENT from the reference is UNVERIFIABLE.
        Do not reward it as richness and do not penalize it as hallucination on its own.
      - Treat surgical synonyms for the SAME action as matching.
      - Mark INCORRECT only if added detail CONTRADICTS the reference, or if the
        prediction substitutes a DIFFERENT action on a structure.

   Decision rule: CORRECT if (a) holds and (b) shows no contradiction or
   meaning-changing substitution. Otherwise INCORRECT.

2. Write a brief assessment (1-3 sentences) explaining your decision.

Respond in EXACTLY this format (no other text):
Verdict: <correct|incorrect>
Assessment: <your explanation>
"""

def _parse_judge_response(text: str) -> tuple:
    verdict = "unknown"
    assessment = text
    m_v = re.search(r"Verdict\s*:\s*(correct|incorrect)", text, re.IGNORECASE)
    m_a = re.search(r"Assessment\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m_v:
        verdict = m_v.group(1).lower()
    if m_a:
        assessment = m_a.group(1).strip()
    return verdict, assessment

JUDGE_MODEL = "claude-opus-4-8"


def validate_judge_config(config):
    if config.get("provider", "anthropic") != "anthropic":
        raise ValueError("Summary judging supports Anthropic only")
    if not config.get("model", JUDGE_MODEL):
        raise ValueError("Set judge.model")
    key_env = config.get("api_key_env", "ANTHROPIC_API_KEY")
    if not os.environ.get(key_env, ""):
        raise ValueError(f"Set {key_env} before enabling summary judging")


def _build_sureon_judge(judge_cfg: dict) -> tuple:
    """Build the paper's Anthropic judge."""
    from anthropic import Anthropic
    client = Anthropic(
        api_key=judge_cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", ""),
        )
    return ("anthropic", judge_cfg.get("model", JUDGE_MODEL), client)


def _ask_judge(judge_tuple: tuple, question: str, ground_truth: str, predicted: str) -> tuple:
    _, model, client = judge_tuple
    prompt = SEQSUM_JUDGE_PROMPT.format(
        question=question, ground_truth=ground_truth, predicted=predicted,
    )
    try:
        response = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return _parse_judge_response(response.content[0].text.strip())
    except Exception as e:
        print(f"Judge request failed: {repr(e)}")
        return ("unknown", "Judge request failed.")


class SummaryJudge:
    def __init__(self, config):
        self.enabled = bool(config.get('enabled'))
        self.judge = None
        if not self.enabled:
            return
        validate_judge_config(config)
        key = os.environ[config.get("api_key_env", "ANTHROPIC_API_KEY")]
        options = dict(config, api_key=key)
        self.judge = _build_sureon_judge(options)

    def evaluate(self, row, prediction):
        if not self.enabled:
            return {'judge_verdict':None,'assessment':None}
        verdict, assessment = _ask_judge(self.judge, row['question'],row['answer_text'],prediction)
        
        if verdict == 'unknown':
            assessment = 'Judge request failed or returned an unparseable verdict.'
        return {'judge_verdict':verdict,'assessment':assessment}
