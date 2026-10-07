"""Usage: python dataproof_cli.py data/orders.csv data/products.csv data/region_summary.csv"""
import sys

from dataproof_engine import answer_question

ICON = {"blocking": "⛔", "warning": "⚠", "info": "ℹ"}


def show(res):
    print("\n" + "=" * 60)
    print("DATA CHECKS")
    for f in res["findings"]:
        print(f"  {ICON[f['severity']]} {f['message']}")
    print("\nSTATUS:", res["status"])
    if res["status"] == "ANSWER":
        print("ANSWER:\n" + res["answer"])
        print("\nCODE:", res["code"])
        for a in res["assumptions"]:
            print("ASSUMPTION:", a)
        print("VERIFIED BY RE-RUN:", "yes" if res["verified"] else "no")
        print("PROOF SCRIPT:", res["proof_script"])
    else:
        print("I can't determine this.\nREASON:", res["reason"])
    print("=" * 60)


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        sys.exit(__doc__)
    while True:
        q = input("\nAsk a question (blank to quit): ").strip()
        if not q:
            break
        show(answer_question(q, paths))
