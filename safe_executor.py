import ast

ALLOWED_METHODS = {
    # aggregation
    "sum", "mean", "max", "min", "count", "median", "nunique", "std", "size",
    "value_counts", "idxmax", "idxmin", "unique", "mode", "prod", "cumsum",
    # selection / shaping
    "head", "tail", "loc", "iloc", "groupby", "agg", "sort_values", "reset_index",
    "set_index", "merge", "drop_duplicates", "dropna", "fillna", "isin", "isna",
    "notna", "between", "astype", "round", "abs", "query", "nlargest", "nsmallest",
    "rename", "copy", "assign", "str", "dt", "lower", "upper", "strip", "contains",
    "year", "month", "day", "quarter", "dayofweek", "values", "shape", "columns",
    "index", "T", "to_numeric", "to_datetime", "Timestamp",
}
ALLOWED_PD = {"to_datetime", "to_numeric", "Timestamp"}
ALLOWED_BUILTINS = {"len": len, "round": round, "abs": abs, "min": min, "max": max,
                    "sum": sum, "float": float, "int": int, "str": str}
BLOCKED_NODES = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Lambda,
                 ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                 ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)


def strip_fences(code):
    code = code.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1] if "\n" in code else ""
        code = code.rsplit("```", 1)[0]
    return code.strip()


def safe_execute(code, tables):
    """Validate and evaluate ONE pandas expression.

    `tables` is a dict {table_name: DataFrame} (a bare DataFrame is treated as `df`).
    """
    import pandas as pd

    if not isinstance(tables, dict):
        tables = {"df": tables}
    code = strip_fences(code)
    tree = ast.parse(code, mode="eval")  # comments are fine in eval mode

    allowed_names = set(tables) | {"pd"} | set(ALLOWED_BUILTINS)

    for node in ast.walk(tree):
        if isinstance(node, BLOCKED_NODES):
            raise ValueError(f"Unsafe code detected: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"Unsafe name detected: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ValueError(f"Unsafe attribute detected: {node.attr}")
            if node.attr not in ALLOWED_METHODS:
                raise ValueError(f"Method not allowed: {node.attr}")
            if isinstance(node.value, ast.Name) and node.value.id == "pd" \
                    and node.attr not in ALLOWED_PD:
                raise ValueError(f"pd.{node.attr} is not allowed")

    env = dict(tables)
    env["pd"] = pd
    return eval(compile(tree, "<safe_code>", "eval"), {"__builtins__": ALLOWED_BUILTINS}, env)


if __name__ == "__main__":
    import pandas as pd
    demo = pd.DataFrame({"Sales": [1, 2, 3], "Region": ["a", "b", "a"]})
    print(safe_execute("df['Sales'].sum()", demo))
    print(safe_execute("df.groupby('Region')['Sales'].sum()", {"df": demo}))
