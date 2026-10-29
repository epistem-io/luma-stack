# %% [markdown]
# # Priority-only ruleset optimization
#
# Constraint: rule *conditions* (thresholds) are fixed. The only lever is the
# `priority` column. This script finds which of the overlaps detected in
# ruleset_robustness_diagnostics.py are structurally fixable by reordering
# (containment) vs. genuinely ambiguous (partial overlap, no forced order)
# vs. unfixable by reordering at all (identical match sets / zero matches).

# %%
import pandas as pd
import numpy as np
from itertools import combinations

VERSION = 'v8'
REGION = 'indonesia'

ruleset_csv = 'C:\\Users\\widijanto\\Documents\\pyprojects\\luma-stack\\data\\modular_mapping_approach\\ruleset_epistem_default_v8.csv'
ruleset = pd.read_csv(ruleset_csv).sort_values('priority').reset_index(drop=True)
synthetic = pd.read_csv('C:\\Users\\widijanto\\Documents\\pyprojects\\luma-stack\\data\\modular_mapping_approach\\synthetic_monte_carlo_v2.csv')

CLASS_NAMES = dict(zip(ruleset['class_id'], ruleset['class_name']))
RULE_COLS = [c for c in ruleset.columns if c not in ('class_id', 'class_name', 'priority')]
current_priority = dict(zip(ruleset['class_id'], ruleset['priority']))

# %% [markdown]
# ## Rule-evaluation logic (verbatim from the DTL script)

# %%
def safe_num(val, default=0):
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return default if np.isnan(val) else float(val)
    try:
        return float(val)
    except Exception:
        return default

def evaluate_condition(row_val, condition_str):
    if condition_str is None:
        return True
    condition_str = str(condition_str).strip()
    if '|' in condition_str:
        return any(evaluate_condition(row_val, c.strip()) for c in condition_str.split('|'))
    if '&' in condition_str:
        return all(evaluate_condition(row_val, c.strip()) for c in condition_str.split('&'))
    if condition_str.startswith('=='):
        op, threshold_str = '==', condition_str[2:]
    elif condition_str.startswith('>='):
        op, threshold_str = '>=', condition_str[2:]
    elif condition_str.startswith('<='):
        op, threshold_str = '<=', condition_str[2:]
    elif condition_str.startswith('>'):
        op, threshold_str = '>', condition_str[1:]
    elif condition_str.startswith('<'):
        op, threshold_str = '<', condition_str[1:]
    else:
        return True
    threshold = safe_num(threshold_str)
    row_val_num = safe_num(row_val)
    if op == '>':
        return row_val_num > threshold
    elif op == '>=':
        return row_val_num >= threshold
    elif op == '<':
        return row_val_num < threshold
    elif op == '<=':
        return row_val_num <= threshold
    elif op == '==':
        return np.isclose(row_val_num, threshold, atol=1e-6)
    return False

def check_rule_match(row, rule):
    for prim in RULE_COLS:
        condition = rule[prim]
        if pd.isna(condition) or condition is None:
            continue
        row_val = row.get(prim, np.nan)
        if not evaluate_condition(row_val, condition):
            return False
    return True

# %% [markdown]
# ## Compute each rule's match vector on a shared synthetic sample
# (order-independent — this ignores priority entirely, just "does this row
# satisfy this rule's conditions")

# %%
N_SAMPLE = 30_000  # containment checks are O(rules^2); keep this moderate
sample = synthetic.sample(n=min(N_SAMPLE, len(synthetic)), random_state=1).reset_index(drop=True)

match_vectors = {}
for _, rule in ruleset.iterrows():
    match_vectors[rule['class_id']] = sample.apply(lambda r: check_rule_match(r, rule), axis=1).to_numpy()

class_ids = ruleset['class_id'].tolist()
raw_match_count = {cid: match_vectors[cid].sum() for cid in class_ids}

never_matches = [cid for cid in class_ids if raw_match_count[cid] == 0]
print("Rules that never match ANY synthetic row — no reordering can fix these,")
print("they need a threshold-level conversation even though thresholds are frozen:")
for cid in never_matches:
    print(f"  - {CLASS_NAMES[cid]} (class_id={cid})")

# %% [markdown]
# ## Classify every overlapping pair: containment / identical / partial

# %%
must_precede_edges = []   # (a, b): a's match set is a strict subset of b's -> a must fire before b
identical_pairs = []      # match sets are exactly the same -> unresolvable via priority
partial_overlap_pairs = []  # share rows but neither contains the other -> needs domain judgment

active_ids = [cid for cid in class_ids if raw_match_count[cid] > 0]

for a, b in combinations(active_ids, 2):
    ma, mb = match_vectors[a], match_vectors[b]
    a_subset_of_b = not np.any(ma & ~mb)
    b_subset_of_a = not np.any(mb & ~ma)
    overlap = np.any(ma & mb)

    if a_subset_of_b and b_subset_of_a:
        identical_pairs.append((a, b))
    elif a_subset_of_b:
        must_precede_edges.append((a, b))  # a narrower, must precede b
    elif b_subset_of_a:
        must_precede_edges.append((b, a))  # b narrower, must precede a
    elif overlap:
        overlap_size = int(np.sum(ma & mb))
        partial_overlap_pairs.append((a, b, overlap_size))

print(f"\nContainment constraints found: {len(must_precede_edges)}")
print(f"Identical match-set pairs (unresolvable via priority): {len(identical_pairs)}")
print(f"Partial-overlap pairs (ambiguous, needs domain judgment): {len(partial_overlap_pairs)}")

# %% [markdown]
# ## Report current-priority violations of containment constraints
# These are unambiguous bugs: reorder to fix, no threshold change needed.

# %%
violations = [
    (a, b) for a, b in must_precede_edges
    if current_priority[a] > current_priority[b]  # a should fire first but currently fires later
]

print(f"\nCurrent priority VIOLATES {len(violations)} containment constraint(s):")
for a, b in violations:
    print(f"  '{CLASS_NAMES[a]}' (priority {current_priority[a]}) is a subset of "
          f"'{CLASS_NAMES[b]}' (priority {current_priority[b]}) but fires AFTER it "
          f"-> '{CLASS_NAMES[a]}' is currently shadowed and effectively dead.")

# %% [markdown]
# ## Propose a reordering: minimal-change topological sort
# Satisfies every containment constraint, breaks ties using the *current*
# priority so the result stays as close as possible to the original intent.

# %%
def suggest_priority_order(class_ids, edges, current_priority):
    # Kahn's algorithm, picking the lowest-current-priority ready node at each step
    successors = {c: [] for c in class_ids}
    indegree = {c: 0 for c in class_ids}
    for a, b in edges:  # a must precede b
        successors[a].append(b)
        indegree[b] += 1

    ready = sorted(
        [c for c in class_ids if indegree[c] == 0],
        key=lambda c: current_priority[c]
    )
    order = []
    indegree = dict(indegree)  # local copy to mutate

    while ready:
        ready.sort(key=lambda c: current_priority[c])
        node = ready.pop(0)
        order.append(node)
        for nxt in successors[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)

    if len(order) != len(class_ids):
        remaining = [c for c in class_ids if c not in order]
        raise RuntimeError(f"Cycle detected among: {[CLASS_NAMES[c] for c in remaining]}")

    return order

suggested_order = suggest_priority_order(class_ids, must_precede_edges, current_priority)

proposal = pd.DataFrame({
    'class_id': suggested_order,
    'class_name': [CLASS_NAMES[c] for c in suggested_order],
    'suggested_priority': range(1, len(suggested_order) + 1),
    'current_priority': [current_priority[c] for c in suggested_order],
})
proposal['moved'] = proposal['suggested_priority'] != proposal['current_priority']

print("\nSuggested priority order (minimal change, all containment constraints satisfied):")
print(proposal.to_string(index=False))

# %% [markdown]
# ## Pairs that reordering CANNOT resolve cleanly — needs domain/ground-truth review

# %%
print("\nIdentical match-set pairs — whichever you rank first wins ALL of these rows,")
print("the other rule becomes permanently unreachable regardless of order:")
for a, b in identical_pairs:
    print(f"  '{CLASS_NAMES[a]}' <-> '{CLASS_NAMES[b]}'")

print("\nPartial-overlap pairs, sorted by overlap size — no forced order exists;")
print("check these specific rows against LULC_24 in the real reference data")
print("to decide which class should take priority in the shared zone:")
for a, b, size in sorted(partial_overlap_pairs, key=lambda x: -x[2])[:15]:
    print(f"  '{CLASS_NAMES[a]}' vs '{CLASS_NAMES[b]}': {size} overlapping rows "
          f"(current priority {current_priority[a]} vs {current_priority[b]})")