import unittest

from limbic.utils.nlp import get_negated_words


TEST_CASES = {
    "I don't like this sentence": ['like'],
    "I won't hate you.": ['hate'],
    'I have not created this.': ['created'],
    "You won't believe me.": ['believe'],
    'Even John could not guess.': ['guess'],
    "no that's not quite it": [],
    "That's not for you to worry about": [],
    "there is not even a dog to give food to": [],
    'This sentence with some random text.': [],
}


class TestNLP(unittest.TestCase):
    def test_process_content(self):
        for input_test, expected_output in TEST_CASES.items():
            output = get_negated_words(input_test)
            self.assertEqual(output, expected_output)
