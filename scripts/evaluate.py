"""
evaluate.py - Evaluation Framework for Multi-Agent Legal RAG System

Compares the multi-agent RAG system against a No-RAG baseline (direct LLM)
to measure the value added by retrieval-augmented generation.

Metrics:
- Groundedness: Does the answer cite specific cases/statutes? (0-1)
- Answer Quality: Assessed based on legal terminology and specificity
- Response Length: Number of words in response
- Response Time: Latency in seconds

Input: eval/RAG_eval_Questions.txt (CSV format with Question,Category)

Outputs:
- JSON results file
- PNG charts (comparison, by category)
- HTML results table

Usage:
    python scripts/evaluate.py [--num-questions N] [--output results.json]
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(override=True)
from langchain_openai import ChatOpenAI

# Configuration
MODEL_NAME = "gpt-4o-mini"
EVAL_DIR = PROJECT_ROOT / "eval"
RESULTS_DIR = PROJECT_ROOT / "evaluation_results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_test_questions(limit: int = None) -> list:
    """Load test questions from the CSV evaluation dataset."""
    questions_path = EVAL_DIR / "RAG_eval_Questions.txt"

    if not questions_path.exists():
        print(f"Error: Test questions not found at {questions_path}")
        sys.exit(1)

    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            question_text = row["Question"].strip().strip('"')
            category = row["Category"].strip()
            questions.append({
                "id": i + 1,
                "question": question_text,
                "category": category,
                "expected_topics": []  # CSV doesn't have predefined topics
            })

    if limit:
        questions = questions[:limit]

    return questions


def run_no_rag_baseline(question: str, llm: ChatOpenAI) -> tuple[str, float]:
    """
    Run direct LLM call without retrieval (No-RAG baseline).

    Returns:
        Tuple of (answer, response_time_seconds)
    """
    start_time = time.time()

    prompt = f"""You are a legal assistant specializing in unemployment insurance law.
Answer the following legal question based on your training knowledge.
Be specific about legal standards, tests, and doctrines.

Question: {question}

Provide a comprehensive answer:"""

    response = llm.invoke(prompt)
    elapsed = time.time() - start_time

    return response.content, elapsed


def run_rag_system(question: str) -> tuple[str, float, list]:
    """
    Run the multi-agent RAG system.

    Returns:
        Tuple of (answer, response_time_seconds, sources)
    """
    from agents.orchestrator import AgentOrchestrator

    start_time = time.time()
    orchestrator = AgentOrchestrator()
    result = orchestrator.run_sync(question)
    elapsed = time.time() - start_time

    # Extract sources from events
    sources = []
    for event in result.get("events", []):
        if event.get("event") == "retrieved_sources":
            sources = event.get("data", {}).get("sources", [])
            break

    return result.get("final_answer", ""), elapsed, sources


def calculate_groundedness(answer: str, sources: list) -> float:
    """
    Calculate groundedness score (0-1).

    Measures whether the answer cites specific cases or legal sources.
    """
    # Check for case citations (e.g., "Smith v. Jones", "123 F.3d 456")
    case_pattern = r'\b[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+'
    citation_pattern = r'\b\d+\s+[A-Z]\.\s*(?:\d+[a-z]*\s+)?\d+'

    case_matches = re.findall(case_pattern, answer)
    citation_matches = re.findall(citation_pattern, answer)

    # Also check if answer references specific statutes
    statute_pattern = r'\b(?:U\.S\.C\.|C\.F\.R\.|Fed\.\s*R\.|Rule\s+\d+|Section\s+\d+|§\s*\d+)'
    statute_matches = re.findall(statute_pattern, answer, re.IGNORECASE)

    total_citations = len(case_matches) + len(citation_matches) + len(statute_matches)

    # Also consider retrieved sources
    has_sources = len(sources) > 0

    if has_sources and total_citations >= 3:
        return 1.0
    elif has_sources or total_citations >= 2:
        return 0.75
    elif total_citations >= 1:
        return 0.5
    elif any(word in answer.lower() for word in ["case", "court", "held", "ruling", "precedent", "statute"]):
        return 0.25
    else:
        return 0.0


def calculate_answer_quality(answer: str) -> float:
    """
    Calculate answer quality score (0-1).

    Measures legal specificity and comprehensiveness.
    """
    score = 0.0
    answer_lower = answer.lower()

    # Check for legal terminology
    legal_terms = [
        "claimant", "eligibility", "benefits", "unemployment", "waiver",
        "recoupment", "overpayment", "fault", "good faith", "agency",
        "administrative", "appeal", "hearing", "determination", "statute"
    ]
    terms_found = sum(1 for term in legal_terms if term in answer_lower)
    score += min(terms_found / 5, 0.4)  # Up to 0.4 for terminology

    # Check for structured reasoning
    if any(marker in answer_lower for marker in ["first", "second", "third", "additionally", "furthermore"]):
        score += 0.2

    # Check for specificity (numbers, dates, sections)
    if re.search(r'\b\d+\b', answer):
        score += 0.2

    # Check for adequate length
    word_count = len(answer.split())
    if word_count >= 100:
        score += 0.2

    return min(score, 1.0)


def evaluate_answer(answer: str, question_data: dict, sources: list = None) -> dict:
    """
    Evaluate a single answer against the expected criteria.
    """
    sources = sources or []

    return {
        "groundedness": calculate_groundedness(answer, sources),
        "answer_quality": calculate_answer_quality(answer),
        "word_count": len(answer.split()),
        "sources_count": len(sources),
    }


def run_evaluation(questions: list, verbose: bool = True) -> dict:
    """
    Run full evaluation comparing RAG system vs No-RAG baseline.
    """
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

    # Get unique categories
    categories = list(set(q["category"] for q in questions))

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "num_questions": len(questions),
        "categories": categories,
        "questions": [],
        "summary": {
            "rag": {"groundedness": 0, "answer_quality": 0, "word_count": 0, "response_time": 0},
            "baseline": {"groundedness": 0, "answer_quality": 0, "word_count": 0, "response_time": 0}
        },
        "by_category": {cat: {"rag": defaultdict(float), "baseline": defaultdict(float), "count": 0} for cat in categories}
    }

    for i, q in enumerate(questions):
        if verbose:
            print(f"\n[{i+1}/{len(questions)}] Evaluating: {q['question'][:60]}...")
            print(f"  Category: {q['category']}")

        question_result = {
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
        }

        # Run baseline (No-RAG)
        if verbose:
            print("  Running baseline (No-RAG)...", end=" ", flush=True)
        try:
            baseline_answer, baseline_time = run_no_rag_baseline(q["question"], llm)
            baseline_scores = evaluate_answer(baseline_answer, q, [])
            baseline_scores["response_time"] = baseline_time
            question_result["baseline"] = {
                "answer": baseline_answer[:500] + "..." if len(baseline_answer) > 500 else baseline_answer,
                "scores": baseline_scores
            }
            if verbose:
                print(f"Done ({baseline_time:.1f}s)")
        except Exception as e:
            print(f"Error: {e}")
            question_result["baseline"] = {"error": str(e), "scores": {}}

        # Run RAG system
        if verbose:
            print("  Running RAG system...", end=" ", flush=True)
        try:
            rag_answer, rag_time, rag_sources = run_rag_system(q["question"])
            rag_scores = evaluate_answer(rag_answer, q, rag_sources)
            rag_scores["response_time"] = rag_time
            question_result["rag"] = {
                "answer": rag_answer[:500] + "..." if len(rag_answer) > 500 else rag_answer,
                "scores": rag_scores,
                "sources": [s.get("case_name", "Unknown") for s in rag_sources[:3]]
            }
            if verbose:
                print(f"Done ({rag_time:.1f}s)")
        except Exception as e:
            print(f"Error: {e}")
            question_result["rag"] = {"error": str(e), "scores": {}}

        results["questions"].append(question_result)

        # Update running totals
        category = q["category"]
        results["by_category"][category]["count"] += 1

        if "scores" in question_result.get("baseline", {}):
            for key in ["groundedness", "answer_quality", "word_count", "response_time"]:
                val = question_result["baseline"]["scores"].get(key, 0)
                results["summary"]["baseline"][key] += val
                results["by_category"][category]["baseline"][key] += val

        if "scores" in question_result.get("rag", {}):
            for key in ["groundedness", "answer_quality", "word_count", "response_time"]:
                val = question_result["rag"]["scores"].get(key, 0)
                results["summary"]["rag"][key] += val
                results["by_category"][category]["rag"][key] += val

    # Calculate averages
    n = len(questions)
    for system in ["rag", "baseline"]:
        for key in results["summary"][system]:
            results["summary"][system][key] /= n

    # Calculate per-category averages
    for cat in categories:
        count = results["by_category"][cat]["count"]
        if count > 0:
            for system in ["rag", "baseline"]:
                for key in results["by_category"][cat][system]:
                    results["by_category"][cat][system][key] /= count
        # Convert defaultdict to regular dict for JSON serialization
        results["by_category"][cat]["rag"] = dict(results["by_category"][cat]["rag"])
        results["by_category"][cat]["baseline"] = dict(results["by_category"][cat]["baseline"])

    return results


def save_visualizations(results: dict, output_dir: Path, timestamp: str):
    """Generate and save visualization charts."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    except ImportError:
        print("Warning: matplotlib not installed. Skipping visualizations.")
        return

    # Chart 1: Overall RAG vs Baseline Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metrics = ["groundedness", "answer_quality"]
    labels = ["Groundedness", "Answer Quality"]
    rag_vals = [results["summary"]["rag"].get(m, 0) for m in metrics]
    base_vals = [results["summary"]["baseline"].get(m, 0) for m in metrics]

    x = range(len(metrics))
    width = 0.35

    axes[0].bar([i - width/2 for i in x], rag_vals, width, label='RAG System', color='#2563eb')
    axes[0].bar([i + width/2 for i in x], base_vals, width, label='No-RAG Baseline', color='#dc2626')
    axes[0].set_ylabel('Score (0-1)')
    axes[0].set_title('RAG vs Baseline: Quality Metrics')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)

    # Response time comparison
    rag_time = results["summary"]["rag"].get("response_time", 0)
    base_time = results["summary"]["baseline"].get("response_time", 0)

    axes[1].bar(['RAG System', 'No-RAG Baseline'], [rag_time, base_time], color=['#2563eb', '#dc2626'])
    axes[1].set_ylabel('Seconds')
    axes[1].set_title('Average Response Time')

    plt.tight_layout()
    chart1_path = output_dir / f"eval_comparison_{timestamp}.png"
    plt.savefig(chart1_path, dpi=150)
    plt.close()
    print(f"Saved: {chart1_path}")

    # Chart 2: Performance by Category
    categories = results.get("categories", [])
    if categories:
        fig, ax = plt.subplots(figsize=(12, 6))

        x = range(len(categories))
        width = 0.35

        rag_ground = [results["by_category"][c]["rag"].get("groundedness", 0) for c in categories]
        base_ground = [results["by_category"][c]["baseline"].get("groundedness", 0) for c in categories]

        ax.bar([i - width/2 for i in x], rag_ground, width, label='RAG System', color='#2563eb')
        ax.bar([i + width/2 for i in x], base_ground, width, label='No-RAG Baseline', color='#dc2626')
        ax.set_ylabel('Groundedness Score')
        ax.set_title('Groundedness by Question Category')
        ax.set_xticks(x)
        ax.set_xticklabels([c[:20] + '...' if len(c) > 20 else c for c in categories], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        chart2_path = output_dir / f"eval_by_category_{timestamp}.png"
        plt.savefig(chart2_path, dpi=150)
        plt.close()
        print(f"Saved: {chart2_path}")


def save_html_table(results: dict, output_dir: Path, timestamp: str):
    """Generate and save HTML results table."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>RAG Evaluation Results - {timestamp}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 2rem; background: #f5f7fa; }}
        h1 {{ color: #1a365d; }}
        h2 {{ color: #2c5282; margin-top: 2rem; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #e2e8f0; padding: 0.75rem; text-align: left; }}
        th {{ background: #1a365d; color: white; }}
        tr:nth-child(even) {{ background: #f7fafc; }}
        .metric {{ font-weight: bold; }}
        .rag {{ color: #2563eb; }}
        .baseline {{ color: #dc2626; }}
        .improvement {{ color: #059669; font-weight: bold; }}
        .summary-card {{ background: white; padding: 1.5rem; border-radius: 8px; margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .category-tag {{ display: inline-block; padding: 0.25rem 0.5rem; background: #e2e8f0; border-radius: 4px; font-size: 0.85rem; margin-right: 0.5rem; }}
    </style>
</head>
<body>
    <h1>Multi-Agent Legal RAG System - Evaluation Results</h1>
    <p><strong>Timestamp:</strong> {results['timestamp']}</p>
    <p><strong>Model:</strong> {results['model']}</p>
    <p><strong>Questions Evaluated:</strong> {results['num_questions']}</p>

    <div class="summary-card">
        <h2>Summary: RAG vs Baseline</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th class="rag">RAG System</th>
                <th class="baseline">No-RAG Baseline</th>
                <th>Improvement</th>
            </tr>
"""

    # Add summary metrics
    metrics = [
        ("Groundedness", "groundedness", True),
        ("Answer Quality", "answer_quality", True),
        ("Word Count", "word_count", False),
        ("Response Time (s)", "response_time", False),
    ]

    for label, key, is_percentage in metrics:
        rag_val = results["summary"]["rag"].get(key, 0)
        base_val = results["summary"]["baseline"].get(key, 0)

        if is_percentage:
            rag_str = f"{rag_val:.1%}"
            base_str = f"{base_val:.1%}"
            if base_val > 0:
                improvement = f"+{((rag_val - base_val) / base_val * 100):.0f}%"
            elif rag_val > 0:
                improvement = "+∞"
            else:
                improvement = "0%"
        else:
            rag_str = f"{rag_val:.1f}"
            base_str = f"{base_val:.1f}"
            improvement = f"{rag_val - base_val:+.1f}"

        html += f"""            <tr>
                <td class="metric">{label}</td>
                <td class="rag">{rag_str}</td>
                <td class="baseline">{base_str}</td>
                <td class="improvement">{improvement}</td>
            </tr>
"""

    html += """        </table>
    </div>

    <h2>Results by Category</h2>
    <table>
        <tr>
            <th>Category</th>
            <th>Questions</th>
            <th>RAG Groundedness</th>
            <th>Baseline Groundedness</th>
        </tr>
"""

    for cat, data in results.get("by_category", {}).items():
        rag_g = data["rag"].get("groundedness", 0)
        base_g = data["baseline"].get("groundedness", 0)
        html += f"""        <tr>
            <td>{cat}</td>
            <td>{data['count']}</td>
            <td class="rag">{rag_g:.1%}</td>
            <td class="baseline">{base_g:.1%}</td>
        </tr>
"""

    html += """    </table>

    <h2>Per-Question Results</h2>
    <table>
        <tr>
            <th>#</th>
            <th>Question</th>
            <th>Category</th>
            <th>RAG Ground.</th>
            <th>Base Ground.</th>
        </tr>
"""

    for q in results["questions"]:
        rag_g = q.get("rag", {}).get("scores", {}).get("groundedness", 0)
        base_g = q.get("baseline", {}).get("scores", {}).get("groundedness", 0)
        question_text = q["question"][:80] + "..." if len(q["question"]) > 80 else q["question"]
        html += f"""        <tr>
            <td>{q['id']}</td>
            <td>{question_text}</td>
            <td><span class="category-tag">{q['category']}</span></td>
            <td class="rag">{rag_g:.0%}</td>
            <td class="baseline">{base_g:.0%}</td>
        </tr>
"""

    html += """    </table>
</body>
</html>
"""

    html_path = output_dir / f"eval_results_{timestamp}.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"Saved: {html_path}")


def print_results_table(results: dict):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nModel: {results['model']}")
    print(f"Questions evaluated: {results['num_questions']}")
    print(f"Timestamp: {results['timestamp']}")

    print("\n" + "-" * 70)
    print(f"{'Metric':<25} {'RAG System':>15} {'No-RAG Baseline':>15} {'Improvement':>12}")
    print("-" * 70)

    rag = results["summary"]["rag"]
    baseline = results["summary"]["baseline"]

    metrics = [
        ("Groundedness", "groundedness", "{:.2%}"),
        ("Answer Quality", "answer_quality", "{:.2%}"),
        ("Avg Word Count", "word_count", "{:.0f}"),
        ("Avg Response Time (s)", "response_time", "{:.1f}"),
    ]

    for label, key, fmt in metrics:
        rag_val = rag.get(key, 0)
        base_val = baseline.get(key, 0)

        if key in ["groundedness", "answer_quality"]:
            if base_val > 0:
                improvement = f"+{((rag_val - base_val) / base_val * 100):.0f}%"
            elif rag_val > 0:
                improvement = "+inf"
            else:
                improvement = "0%"
        elif key == "response_time":
            improvement = f"{base_val - rag_val:.1f}s" if base_val > rag_val else f"+{rag_val - base_val:.1f}s"
        else:
            improvement = f"+{rag_val - base_val:.0f}"

        print(f"{label:<25} {fmt.format(rag_val):>15} {fmt.format(base_val):>15} {improvement:>12}")

    print("-" * 70)

    # Per-category breakdown
    print("\nResults by Category:")
    print("-" * 70)
    for cat, data in results.get("by_category", {}).items():
        rag_g = data["rag"].get("groundedness", 0)
        base_g = data["baseline"].get("groundedness", 0)
        print(f"  {cat}: RAG={rag_g:.0%} vs Base={base_g:.0%} ({data['count']} questions)")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Agent Legal RAG System")
    parser.add_argument("--num-questions", "-n", type=int, default=None,
                        help="Number of questions to evaluate (default: all)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file for results (default: auto-generated)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip generating visualization charts")

    args = parser.parse_args()

    print("=" * 70)
    print("Multi-Agent Legal RAG System - Evaluation")
    print("=" * 70)
    print(f"\nComparing RAG System vs No-RAG Baseline")
    print(f"Model: {MODEL_NAME}")

    # Load questions
    questions = load_test_questions(args.num_questions)
    print(f"Loaded {len(questions)} test questions from eval/RAG_eval_Questions.txt")

    # Show categories
    categories = list(set(q["category"] for q in questions))
    print(f"Categories: {', '.join(categories)}")

    # Run evaluation
    results = run_evaluation(questions, verbose=not args.quiet)

    # Print results
    print_results_table(results)

    # Generate timestamp for all outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON results
    output_path = args.output
    if not output_path:
        output_path = RESULTS_DIR / f"eval_results_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved to: {output_path}")

    # Generate visualizations
    if not args.no_charts:
        save_visualizations(results, RESULTS_DIR, timestamp)

    # Generate HTML table
    save_html_table(results, RESULTS_DIR, timestamp)

    print(f"\nAll outputs saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
