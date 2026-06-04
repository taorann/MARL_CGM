from __future__ import annotations

import re

TEST_PATH_RE = re.compile(r"(^|/)(tests?|testing|test_[^/]*\.py|[^/]*_test\.py)(/|$)")


def is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return bool(TEST_PATH_RE.search(normalized))
