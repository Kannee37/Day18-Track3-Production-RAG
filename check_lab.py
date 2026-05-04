"""Submission checks for Lab 18."""

import json
import os
import re
import subprocess
import sys


def check_file(path: str, required: bool = True) -> bool:
    if os.path.exists(path):
        print(f"  OK: {path}")
        return True
    if required:
        print(f"  MISSING: {path}")
        return False
    print(f"  Optional missing: {path}")
    return True


def check_json(path: str, required_keys: list[str]) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        missing = [key for key in required_keys if key not in data]
        if missing:
            print(f"  BAD: {path} missing keys: {missing}")
            return False
        print(f"  OK: {path} keys OK")
        return True
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        print(f"  BAD: {path} - {exc}")
        return False


def check_todos() -> int:
    count = 0
    for root, _, files in os.walk("src"):
        for filename in files:
            if filename.endswith(".py"):
                with open(os.path.join(root, filename), encoding="utf-8") as fh:
                    for line in fh:
                        if "# TODO:" in line:
                            count += 1
    return count


def run_tests() -> tuple[int, int]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=no", "-q"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = f"{result.stdout}\n{result.stderr}"
        passed = total = 0
        passed_match = re.search(r"(\d+)\s+passed", output)
        failed_match = re.search(r"(\d+)\s+failed", output)
        if passed_match:
            passed = int(passed_match.group(1))
            total += passed
        if failed_match:
            total += int(failed_match.group(1))
        return passed, total
    except Exception as exc:
        print(f"  pytest error: {exc}")
        return 0, 0


def validate():
    print("Checking Lab 18 submission\n")
    errors = 0

    print("Source code:")
    for path in ["src/m1_chunking.py", "src/m2_search.py", "src/m3_rerank.py", "src/m4_eval.py", "src/pipeline.py"]:
        if not check_file(path):
            errors += 1

    print("\nReports:")
    if check_file("reports/ragas_report.json"):
        if not check_json("reports/ragas_report.json", ["aggregate", "num_questions"]):
            errors += 1
    else:
        errors += 1
    check_file("reports/naive_baseline_report.json", required=False)

    print("\nAnalysis:")
    check_file("analysis/failure_analysis.md")
    check_file("analysis/group_report.md")

    print("\nIndividual reflections:")
    ref_dir = "analysis/reflections"
    reflections = []
    if os.path.isdir(ref_dir):
        reflections = [f for f in os.listdir(ref_dir) if f.startswith("reflection_") and f.endswith(".md")]
    if reflections:
        for reflection in reflections:
            print(f"  OK: {ref_dir}/{reflection}")
    else:
        print(f"  Optional missing: {ref_dir}/reflection_*.md")

    print("\nTODO markers:")
    todo_count = check_todos()
    if todo_count == 0:
        print("  OK: no TODO markers")
    else:
        print(f"  WARN: {todo_count} TODO markers remain")

    print("\nAuto-tests:")
    passed, total = run_tests()
    if total > 0:
        pct = passed / total * 100
        print(f"  {'OK' if pct >= 80 else 'WARN'}: {passed}/{total} tests passed ({pct:.0f}%)")
    else:
        print("  WARN: tests did not run")

    print("\n" + "=" * 50)
    if errors == 0:
        print("Submission structure is ready.")
    else:
        print(f"There are {errors} required-file/report errors.")
    print("=" * 50)


if __name__ == "__main__":
    validate()
