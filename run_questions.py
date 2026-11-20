"""
run_questions.py - Assessment Runner for AeroVantage X9 Drone Manual RAG.

Executes all 5 required assessment questions, records answers and sources,
outputs structured JSON and Markdown to outputs/, and displays a portal-ready block.
"""

import json
from pathlib import Path
from rag import answer

ASSESSMENT_QUESTIONS = [
    "What does the diagnostic LED pattern of three rapid red flashes followed by two solid amber flashes indicate on the X9 flight controller?",
    "Can the AeroVantage X9 be used for underwater sonar mapping at a depth of 50 metres?",
    "At what minimum per-cell voltage, in volts, does the X9 trigger a forced emergency Return-to-Home (RTH) landing?",
    "What is the maximum sustained wind speed at which the X9 is certified to operate safely?",
    "Who is the current Chief Executive Officer of DJI Innovations?"
]


def run_all_questions():
    outputs_dir = Path("./outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print("=" * 80)
    print("RUNNING ASSESSMENT QUESTIONS ON AEROVANTAGE X9 RAG PIPELINE")
    print("=" * 80)

    for idx, q in enumerate(ASSESSMENT_QUESTIONS, start=1):
        q_id = f"Q{idx}"
        print(f"\nProcessing [{q_id}]: {q}")
        res = answer(q)
        ans_text = res.get("answer", "[not found]")
        sources = res.get("sources", [])
        print(f"  -> Answer:  {ans_text}")
        print(f"  -> Sources: {sources}")

        results.append({
            "id": q_id,
            "question": q,
            "answer": ans_text,
            "sources": sources
        })

    # 1. Save JSON output
    json_path = outputs_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved structured results to: {json_path}")

    # 2. Save Markdown output
    md_path = outputs_dir / "results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# AeroVantage X9 Drone Manual RAG - Assessment Results\n\n")
        f.write("| ID | Question | Answer | Source Pages |\n")
        f.write("|---|---|---|---|\n")
        for r in results:
            src_str = ", ".join(str(p) for p in r["sources"]) if r["sources"] else "None"
            # Sanitize pipe symbols in answer
            clean_ans = r["answer"].replace("|", "\\|")
            f.write(f"| **{r['id']}** | {r['question']} | {clean_ans} | {src_str} |\n")

        f.write("\n## Portal-Ready Output Block\n\n```text\n")
        f.write("PROJECT NAME: Industrial Drone & Hardware Technical Manual Assistant\n")
        for r in results:
            f.write(f"{r['id']}: {r['answer']} | sources: {r['sources']}\n")
        f.write("```\n")
    print(f"Saved Markdown report to: {md_path}")

    # 3. Print Portal-Ready Block for Copy-Pasting
    print("\n" + "=" * 80)
    print("PORTAL-READY COPY-PASTE BLOCK:")
    print("=" * 80)
    print("PROJECT NAME: Industrial Drone & Hardware Technical Manual Assistant")
    for r in results:
        print(f"{r['id']}: {r['answer']} | sources: {r['sources']}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_all_questions()
