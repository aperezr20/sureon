import re

def _extract_answer(text: str) -> str:
    m = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    return m.group(1).strip() if m else text

def _strip_list_prefix(text: str) -> str:
    """Remove leading markdown list markers (- , * , • ) a model may prepend."""
    return re.sub(r"^[-*•]\s+", "", text)

def _normalize_temporal_ordering(s: str) -> list:
    s = s.replace(" ", "")
    letters = re.findall(r"\b([A-Za-z])\b", s)
    return [l.upper() for l in letters]


def score_answer(row, prediction):

    if row['question_type'] == 'sequence_summarization': # summaries require LLM Judge
        return None
    
    truth = _extract_answer(row['answer_text'])
    predicted = _extract_answer(prediction)
    
    if row['question_type'] == 'temporal_ordering':
        return _normalize_temporal_ordering(predicted) == _normalize_temporal_ordering(truth)
    
    return _strip_list_prefix(predicted).lower() == _strip_list_prefix(truth).lower()


def metrics(records):
    from collections import defaultdict
    
    accuracy = [r for r in records if r['question_type'] != 'sequence_summarization']
    summaries = [r for r in records if r['question_type'] == 'sequence_summarization']
    groups = defaultdict(list)
    
    for row in accuracy:
        groups[row['question_type']].append(row)
    
    def aggregate(rows):
        n = len(rows)
        correct = sum(r.get('correct') is True for r in rows)
        completed = sum(r['status'] == 'ok' for r in rows)
        return {'correct': correct, 'expected': n, 'completed': completed,
                'accuracy': correct / n if n else None}
    
    by_type = {key:aggregate(rows) for key,rows in sorted(groups.items())}
    judged = [r for r in summaries if r.get('judge_verdict') in {'correct','incorrect'}]
    summary_correct = sum(r.get('judge_verdict') == 'correct' for r in judged)
    
    return {'accuracy_questions':aggregate(accuracy), 'per_category':by_type,
            'mean_category_accuracy':sum(v['accuracy'] for v in by_type.values()) / len(by_type) if by_type else None,
            'sequence_summarization':{'expected':len(summaries),'judged':len(judged),
                'correct':summary_correct,
                'accuracy':summary_correct / len(summaries) if summaries and len(judged)==len(summaries) else None},
            'failed_questions':sum(r['status']!='ok' for r in records),
            'judge_failures':sum(r.get('judge_verdict')=='unknown' for r in summaries)}
