import ast
import hashlib
import json
from pathlib import Path


TERM_NAMES = ("a_B1", "a_B2", "a_M1", "a_M2", "a_m", "a")
MANIFEST_PATH = Path("docs/provenance/weak_form_manifest.json")
TARGET_PATHS = (
    "src/lung_pde/forms/baseline.py",
    "src/lung_pde/forms/calibration.py",
    "src/lung_pde/forms/velocity.py",
    "src/lung_pde/forms/treatment.py",
)
EXPECTED_MANIFEST = [
    {
        "fixture": "docs/provenance/legacy_form_blocks/test_final.py",
        "source_path": "/Users/wangjindong/Desktop/lung/test_final.py",
        "line_range": [168, 186],
        "sha256": "2b68c862f60a143fb3fef1f417bf672c62f0814209041f9293953cf5c5869c32",
        "term_names": list(TERM_NAMES),
        "targets": [TARGET_PATHS[0], TARGET_PATHS[1]],
    },
    {
        "fixture": "docs/provenance/legacy_form_blocks/test_v4.py",
        "source_path": "/Users/wangjindong/Desktop/lung/test-v4.py",
        "line_range": [182, 200],
        "sha256": "2353a6a882b761642d9bec1994434dd40ad8ed13d050fc6a391e54f65eeb053d",
        "term_names": list(TERM_NAMES),
        "targets": [TARGET_PATHS[2]],
    },
    {
        "fixture": "docs/provenance/legacy_form_blocks/test_anti.py",
        "source_path": "/Users/wangjindong/Desktop/lung/test_anti.py",
        "line_range": [179, 197],
        "sha256": "a93d0afe16d958b6e08d7284fd8a820b66493d8cf8503487d8a4ffc51e989ee5",
        "term_names": list(TERM_NAMES),
        "targets": [TARGET_PATHS[3]],
    },
]


def _manifest():
    return json.loads(MANIFEST_PATH.read_text())


def _single_name_assignment(statement):
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _protected_assignments(statements):
    indexed = [
        (index, statement)
        for index, statement in enumerate(statements)
        if _single_name_assignment(statement) in TERM_NAMES
    ]
    names = tuple(_single_name_assignment(statement) for _, statement in indexed)
    assert names == TERM_NAMES
    indices = tuple(index for index, _ in indexed)
    assert indices == tuple(range(indices[0], indices[0] + len(TERM_NAMES)))
    return tuple(statement for _, statement in indexed)


def _fixture_assignments(path: Path):
    statements = ast.parse(path.read_text()).body
    assignments = _protected_assignments(statements)
    assert len(statements) == len(TERM_NAMES)
    return assignments


def _build_form(path: Path):
    matches = [
        statement
        for statement in ast.parse(path.read_text()).body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name == "build_form"
    ]
    assert len(matches) == 1
    return matches[0]


def _module_assignments(path: Path):
    return _protected_assignments(_build_form(path).body)


def _assignment_trees(assignments):
    return tuple(
        ast.dump(statement.value, include_attributes=False)
        for statement in assignments
    )


def _multiplication_count(path: Path, left_name: str, right_name: str) -> int:
    tree = ast.parse(path.read_text())
    return sum(
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mult)
        and isinstance(node.left, ast.Name)
        and node.left.id == left_name
        and isinstance(node.right, ast.Name)
        and node.right.id == right_name
        for node in ast.walk(tree)
    )


def _is_avg_difference(node: ast.AST, upper: str, lower: str) -> bool:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Sub):
        return False
    expected = ast.parse(f"avg({upper}) - avg({lower})", mode="eval").body
    return ast.dump(node, include_attributes=False) == ast.dump(
        expected, include_attributes=False
    )


def test_manifest_has_complete_unique_form_coverage():
    manifest = _manifest()
    assert manifest == EXPECTED_MANIFEST

    targets = [target for entry in manifest for target in entry["targets"]]
    assert len(targets) == len(set(targets))
    assert set(targets) == set(TARGET_PATHS)

    executable_modules = {
        str(path)
        for path in Path("src/lung_pde/forms").glob("*.py")
        if path.name != "__init__.py"
    }
    assert executable_modules == set(TARGET_PATHS)


def test_form_fixtures_and_build_form_blocks_match_exactly():
    for entry in EXPECTED_MANIFEST:
        fixture = Path(entry["fixture"])
        assert hashlib.sha256(fixture.read_bytes()).hexdigest() == entry["sha256"]
        expected = _assignment_trees(_fixture_assignments(fixture))
        for target in entry["targets"]:
            assert _assignment_trees(_module_assignments(Path(target))) == expected


def test_baseline_and_calibration_keep_scaled_interface_coefficients():
    for target in map(Path, TARGET_PATHS[:2]):
        assert _multiplication_count(target, "DB1", "tauB") == 1
        assert _multiplication_count(target, "DB2", "tauB") == 1
        assert _multiplication_count(target, "DM", "tauM") == 2


def test_executable_forms_exclude_unscaled_interface_assignments():
    for target in map(Path, TARGET_PATHS):
        tree = ast.parse(target.read_text())
        for node in ast.walk(tree):
            assert not (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Mult)
                and isinstance(node.left, ast.Name)
                and node.left.id == "tauB"
                and _is_avg_difference(node.right, "B2", "B1")
            )
            assert not (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Mult)
                and isinstance(node.left, ast.Name)
                and node.left.id == "tauM"
                and _is_avg_difference(node.right, "M2", "M1")
            )


def test_jacobians_and_form_bundle_returns_are_exact_and_follow_blocks():
    expected_jacobian = ast.parse(
        "derivative(a, w, TrialFunction(w.function_space()))", mode="eval"
    ).body
    expected_return = ast.parse(
        "FormBundle(a, J, (a_B1, a_B2, a_M1, a_M2, a_m))", mode="eval"
    ).body

    for target in map(Path, TARGET_PATHS):
        statements = _build_form(target).body
        protected = _protected_assignments(statements)
        protected_end = statements.index(protected[-1])

        jacobians = [
            (index, statement)
            for index, statement in enumerate(statements)
            if _single_name_assignment(statement) == "J"
        ]
        assert len(jacobians) == 1
        jacobian_index, jacobian = jacobians[0]
        assert jacobian_index > protected_end
        assert ast.dump(jacobian.value, include_attributes=False) == ast.dump(
            expected_jacobian, include_attributes=False
        )

        returns = [
            (index, statement)
            for index, statement in enumerate(statements)
            if isinstance(statement, ast.Return)
        ]
        assert len(returns) == 1
        return_index, returned = returns[0]
        assert return_index > jacobian_index
        assert ast.dump(returned.value, include_attributes=False) == ast.dump(
            expected_return, include_attributes=False
        )


def test_velocity_mucus_advection_keeps_test_gradient_sign_and_state_order():
    assignments = {
        _single_name_assignment(statement): statement
        for statement in _module_assignments(Path(TARGET_PATHS[2]))
    }

    for term, test_function, state in (
        ("a_B1", "E1", "B1"),
        ("a_M1", "F1", "M1"),
        ("a_m", "s", "m"),
    ):
        expected = ast.parse(
            f"dot(u, grad({test_function})) * {state} * dx(MUCUS_FLAG)",
            mode="eval",
        ).body
        expected_tree = ast.dump(expected, include_attributes=False)
        matching_nodes = [
            node
            for node in ast.walk(assignments[term].value)
            if isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Sub)
            and ast.dump(node.right, include_attributes=False) == expected_tree
        ]
        assert len(matching_nodes) == 1
