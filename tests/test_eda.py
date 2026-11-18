from eda_sroie import TOTAL_WORDS, shape


def test_shape_maps_digits_and_letters():
    assert shape("25/12/2018") == "99/99/9999"
    assert shape("12 JAN 2018") == "99 AAA 9999"
    assert shape("RM9.00") == "AA9.99"
    assert shape("") == ""


def test_total_keywords_match_whole_words_only():
    assert TOTAL_WORDS.search("SUB TOTAL 12.00")
    assert TOTAL_WORDS.search("Rounding Adj")
    assert not TOTAL_WORDS.search("TOTALLY FRESH")
