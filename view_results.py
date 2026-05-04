import pickle
from pprint import pprint

path = "/home/pjh/clone_repo/OVTR/ovtr/results/teta_results_lite_qat_exp_a2_val/OVTR/teta_summary_results.pth"

with open(path, "rb") as f:
    res = pickle.load(f)

fields = [
    "TETA", "LocA", "AssocA", "ClsA",
    "LocRe", "LocPr",
    "AssocRe", "AssocPr",
    "ClsRe", "ClsPr",
]

# 전체 summary 위치
overall = res["COMBINED_SEQ"]["average"]["TETA"]

print("available keys:", overall.keys())
print("\n[50]")
pprint(dict(zip(fields, overall[50])))

print("\n[ALL]")
pprint(dict(zip(fields, overall["ALL"])))