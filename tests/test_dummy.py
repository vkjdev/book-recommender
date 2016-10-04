from recommender import Dummy


def test_dummy():
    dummy = Dummy()
    assert dummy.dummy_method() == True
