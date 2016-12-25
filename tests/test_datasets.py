import pytest

from questionanswering.datasets import webquestions_io


def test_load_webquestions():
    webquestions = webquestions_io.WebQuestions({})


if __name__ == '__main__':
    pytest.main([__file__])
