from codesearch.eval.metrics import macro_average, mrr, precision_at_k, recall_at_k


def test_precision_at_k_all_relevant():
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0


def test_precision_at_k_none_relevant():
    assert precision_at_k(["a", "b", "c"], {"x", "y"}, 3) == 0.0


def test_precision_at_k_partial():
    # top-3 has 1 relevant ("b") out of 3 -> 1/3
    assert precision_at_k(["a", "b", "c"], {"b"}, 3) == 1 / 3


def test_precision_at_k_respects_k_cutoff():
    # relevant item is at rank 4, outside k=2 -> 0/2
    assert precision_at_k(["a", "b", "c", "d"], {"d"}, 2) == 0.0


def test_precision_at_k_empty_retrieved():
    assert precision_at_k([], {"a"}, 5) == 0.0


def test_recall_at_k_finds_all():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == 1.0


def test_recall_at_k_partial():
    # only "a" found out of 2 relevant -> 1/2
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5


def test_recall_at_k_respects_k_cutoff():
    # relevant item at rank 3, outside k=1 -> 0/1
    assert recall_at_k(["x", "y", "a"], {"a"}, 1) == 0.0


def test_recall_at_k_no_relevant_items_is_zero():
    assert recall_at_k(["a", "b"], set(), 5) == 0.0


def test_recall_at_k_never_exceeds_one_with_duplicate_matches():
    # Two distinct retrieved chunks (e.g. a class summary + a method chunk)
    # can both match the same single relevant item - recall must cap at
    # 1.0, not count each occurrence separately.
    assert recall_at_k(["a", "a", "a"], {"a"}, 3) == 1.0


def test_mrr_first_rank():
    assert mrr(["a", "b", "c"], {"a"}) == 1.0


def test_mrr_third_rank():
    assert mrr(["x", "y", "a"], {"a"}) == 1 / 3


def test_mrr_no_hit():
    assert mrr(["x", "y", "z"], {"a"}) == 0.0


def test_mrr_uses_first_hit_only():
    # "a" at rank 1, "b" at rank 3 -> should score on rank 1, not rank 3
    assert mrr(["a", "x", "b"], {"a", "b"}) == 1.0


def test_macro_average():
    assert macro_average([1.0, 0.5, 0.0]) == 0.5


def test_macro_average_empty():
    assert macro_average([]) == 0.0
