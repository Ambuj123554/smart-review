#!/usr/bin/env python3
"""
CLI Test Script for Smart Code Review Assistant.

Usage:
    python test_review.py
    python test_review.py sample_code/01_mutable_default.py
    python test_review.py --all
"""

import sys
import os
import glob
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.analyzer.engine import ASTAnalyzer
from app.ai.reviewer import GeminiReviewer
from app.database.db import DatabaseManager
from app.schemas.review import ReviewRequest, ReviewResponse, Finding


# Reconfigure stdout to use UTF-8 on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ANSI Color Codes for clean terminal output
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_CYAN = "\033[96m"
COLOR_GREEN = "\033[92m"

SEVERITY_COLORS = {
    "CRITICAL": COLOR_RED + COLOR_BOLD,
    "HIGH": COLOR_RED,
    "MEDIUM": COLOR_YELLOW,
    "LOW": COLOR_CYAN
}


def print_banner():
    print(f"\n{COLOR_BOLD}{COLOR_BLUE}==============================================================={COLOR_RESET}")
    print(f"{COLOR_BOLD}{COLOR_BLUE}         SMART CODE REVIEW ASSISTANT - CLI DEMO                {COLOR_RESET}")
    print(f"{COLOR_BOLD}{COLOR_BLUE}==============================================================={COLOR_RESET}\n")


def print_finding(finding: Finding, index: int):
    sev_color = SEVERITY_COLORS.get(finding.severity.upper(), COLOR_RESET)
    source_badge = f"[{finding.source}]"
    
    print(f"  {COLOR_BOLD}Finding #{index}{COLOR_RESET} | {COLOR_BOLD}Line {finding.line}{COLOR_RESET}")
    print(f"    |- Rule ID     : {COLOR_CYAN}{finding.rule_id}{COLOR_RESET}")
    print(f"    |- Category    : {finding.category}")
    print(f"    |- Severity    : {sev_color}{finding.severity}{COLOR_RESET}")
    print(f"    |- Source      : {source_badge}")
    print(f"    |- Issue       : {finding.message}")
    print(f"    |- Explanation : {finding.explanation}")
    print(f"    \\_ Suggestion  : {COLOR_GREEN}{finding.suggestion}{COLOR_RESET}\n")



def review_file(file_path: str, ast_analyzer: ASTAnalyzer, ai_reviewer: GeminiReviewer, db_manager: DatabaseManager):
    print(f"{COLOR_BOLD}{COLOR_BLUE}---------------------------------------------------------------{COLOR_RESET}")
    print(f"Reviewing File: {COLOR_BOLD}{file_path}{COLOR_RESET}")
    print(f"{COLOR_BOLD}{COLOR_BLUE}---------------------------------------------------------------{COLOR_RESET}")

    if not os.path.exists(file_path):
        print(f"{COLOR_RED}Error: File '{file_path}' does not exist.{COLOR_RESET}\n")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    # 1. AST Analysis
    rule_findings = ast_analyzer.analyze(code)

    # 2. AI Analysis
    ai_findings = ai_reviewer.analyze(code, rule_findings)

    all_findings = rule_findings + ai_findings

    # Count Severities
    crit = sum(1 for f in all_findings if f.severity.upper() == "CRITICAL")
    high = sum(1 for f in all_findings if f.severity.upper() == "HIGH")
    med = sum(1 for f in all_findings if f.severity.upper() == "MEDIUM")
    low = sum(1 for f in all_findings if f.severity.upper() == "LOW")

    # Print Summary Table
    print(f"\n{COLOR_BOLD}SUMMARY OF FINDINGS:{COLOR_RESET}")
    print(f"  Total Issues : {COLOR_BOLD}{len(all_findings)}{COLOR_RESET}")
    print(f"  Critical     : {COLOR_RED}{crit}{COLOR_RESET}")
    print(f"  High         : {COLOR_RED}{high}{COLOR_RESET}")
    print(f"  Medium       : {COLOR_YELLOW}{med}{COLOR_RESET}")
    print(f"  Low          : {COLOR_CYAN}{low}{COLOR_RESET}")
    print(f"  Rule Findings: {len(rule_findings)}")
    print(f"  AI Findings  : {len(ai_findings)}\n")

    if not all_findings:
        print(f"  {COLOR_GREEN}[OK] No issues detected! Code looks clean.{COLOR_RESET}\n")
        return

    print(f"{COLOR_BOLD}DETAILED FINDINGS:{COLOR_RESET}\n")
    for idx, finding in enumerate(all_findings, start=1):
        print_finding(finding, idx)


def main():
    print_banner()

    ast_analyzer = ASTAnalyzer()
    ai_reviewer = GeminiReviewer()
    db_manager = DatabaseManager()

    # Determine target files
    if len(sys.argv) > 1:
        target_arg = sys.argv[1]
        if target_arg in ("--all", "-a"):
            files = sorted(glob.glob("sample_code/*.py"))
        else:
            files = [target_arg]
    else:
        # Default: run through all sample code files
        files = sorted(glob.glob("sample_code/*.py"))

    if not files:
        print(f"{COLOR_YELLOW}No Python sample files found in sample_code/.{COLOR_RESET}")
        return

    print(f"Found {len(files)} file(s) for review.\n")
    for file_path in files:
        review_file(file_path, ast_analyzer, ai_reviewer, db_manager)

    print(f"{COLOR_BOLD}{COLOR_GREEN}[OK] Demonstration review completed successfully.{COLOR_RESET}\n")



if __name__ == "__main__":
    main()
