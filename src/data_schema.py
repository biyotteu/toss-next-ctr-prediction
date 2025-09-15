import re
from typing import List

# Patterns to group columns
CAT_PATTERNS = [
    r"^gender$", r"^age_group$", r"^inventory_id$",
    r"^l_feat_\d+$", r"^feat_[a-e]_\d+$", r"^history_a_\d+$",
    r"^day_of_week$", r"^hour$",
]

EXCLUDE_COLS = set(["clicked", "ID", "seq"])  # handled separately

# is categorical by name or dtype('O') decided later

def match_any(name: str, patterns: List[str]) -> bool:
    return any(re.match(p, name) for p in patterns)