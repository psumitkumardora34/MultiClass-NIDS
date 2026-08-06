"""
model_compat.py
===============
Compatibility patch for sklearn 1.2.2 models running on sklearn 1.3+.

Issues fixed:
1. DecisionTree node array dtype missing 'missing_go_to_left' field
   - sklearn 1.2.2 → 1.4.x : 7-field NODE_DTYPE (no missing_go_to_left)
   - sklearn 1.5+           : 8-field NODE_DTYPE (with missing_go_to_left)
2. DecisionTreeClassifier / ExtraTreeRegressor missing 'monotonic_cst'
3. IsolationForest missing '_decision_path_lengths' and
   '_average_path_length_per_tree' (added in sklearn 1.5)
4. Tree object missing 'compute_node_depths()' method (added in sklearn 1.3)

Apply this module before loading ANY model pickle file.
"""

import numpy as np
import sklearn
import sklearn.tree._tree as _tree_module

# ---------------------------------------------------------------------------
# Version info
# ---------------------------------------------------------------------------
_sklearn_ver = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
_NEW_NODE_DTYPE = _tree_module.NODE_DTYPE
_HAS_MISSING_GO_TO_LEFT = 'missing_go_to_left' in _NEW_NODE_DTYPE.names
_HAS_COMPUTE_NODE_DEPTHS = hasattr(_tree_module.Tree, 'compute_node_depths')

# Old 7-field dtype from sklearn 1.2.2
_OLD_NODE_DTYPE = np.dtype({
    'names': [
        'left_child', 'right_child', 'feature', 'threshold',
        'impurity', 'n_node_samples', 'weighted_n_node_samples'
    ],
    'formats': ['<i8', '<i8', '<i8', '<f8', '<f8', '<i8', '<f8'],
    'offsets': [0, 8, 16, 24, 32, 40, 48],
    'itemsize': 56
})
_COMMON_FIELDS = list(_OLD_NODE_DTYPE.names)

TREE_LEAF = _tree_module.TREE_LEAF

# ---------------------------------------------------------------------------
# 1. Patch _check_node_ndarray (handles both 7-field and 8-field targets)
# ---------------------------------------------------------------------------
_orig_check = _tree_module._check_node_ndarray


def _patched_check_node_ndarray(node_ndarray, expected_dtype):
    if node_ndarray.dtype == _OLD_NODE_DTYPE and node_ndarray.dtype != expected_dtype:
        new_arr = np.zeros(node_ndarray.shape, dtype=expected_dtype)
        for name in _COMMON_FIELDS:
            if name in expected_dtype.names:
                new_arr[name] = node_ndarray[name]
        # Only set missing_go_to_left when the target dtype actually has it
        if _HAS_MISSING_GO_TO_LEFT and 'missing_go_to_left' in expected_dtype.names:
            new_arr['missing_go_to_left'] = 1
        return new_arr
    return _orig_check(node_ndarray, expected_dtype)


_tree_module._check_node_ndarray = _patched_check_node_ndarray

print(f"[COMPAT] sklearn {sklearn.__version__} detected — "
      f"node dtype patch applied "
      f"({'8-field' if _HAS_MISSING_GO_TO_LEFT else '7-field'} mode)")


# ---------------------------------------------------------------------------
# 2. Manual node-depth computation
#    Replaces tree.compute_node_depths() which was added in sklearn 1.3.
# ---------------------------------------------------------------------------
def _compute_node_depths(tree) -> np.ndarray:
    """
    Return depth of every node in the tree (root = depth 0).
    Works on any sklearn version — does NOT call compute_node_depths().
    """
    n_nodes = tree.node_count
    depths = np.zeros(n_nodes, dtype=np.intp)
    stack = [(0, 0)]  # (node_id, depth)
    while stack:
        node_id, depth = stack.pop()
        depths[node_id] = depth
        left  = tree.children_left[node_id]
        right = tree.children_right[node_id]
        if left != TREE_LEAF:
            stack.append((left,  depth + 1))
            stack.append((right, depth + 1))
    return depths


# ---------------------------------------------------------------------------
# 3 & 4. Patch loaded estimators after joblib.load()
# ---------------------------------------------------------------------------
def patch_rf(rf_model):
    """
    Patch a RandomForestClassifier loaded from sklearn 1.2.2.
    Adds missing attributes to each DecisionTree estimator.
    """
    for est in rf_model.estimators_:
        if not hasattr(est, 'monotonic_cst'):
            est.monotonic_cst = None
        if not hasattr(est, 'n_features_'):
            est.n_features_ = est.n_features_in_
    return rf_model


def patch_iso(iso_model):
    """
    Patch an IsolationForest loaded from sklearn 1.2.2.

    - Adds missing tree-level attributes (monotonic_cst, n_features_).
    - Recomputes _decision_path_lengths and _average_path_length_per_tree
      without calling compute_node_depths() (not available in 1.2.2).
    """
    from sklearn.ensemble._iforest import _average_path_length

    for est in iso_model.estimators_:
        if not hasattr(est, 'monotonic_cst'):
            est.monotonic_cst = None
        if not hasattr(est, 'n_features_'):
            est.n_features_ = est.n_features_in_

    if (not hasattr(iso_model, '_decision_path_lengths') or
            not hasattr(iso_model, '_average_path_length_per_tree')):

        avg_path_lengths_list = []
        decision_path_lengths_list = []

        for est in iso_model.estimators_:
            tree = est.tree_

            # Node depths — use manual BFS, never compute_node_depths()
            node_depths = _compute_node_depths(tree)

            # Average path length per node (indexed by node id)
            node_avg_path = _average_path_length(tree.n_node_samples)

            avg_path_lengths_list.append(node_avg_path)
            decision_path_lengths_list.append(node_depths)

        iso_model._average_path_length_per_tree = avg_path_lengths_list
        iso_model._decision_path_lengths = decision_path_lengths_list

    return iso_model
