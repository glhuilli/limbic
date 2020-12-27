import unittest

from limbic.utils.text import process_content


TEST_CASES = {
    'Limbic is a package.': ['limbic', 'package'],
    'a random number 111': ['random', 'number'],
    'string with font color': [''],  # hardcoded on purpose for subtitle analysis TODO: fix this
    'anything @ something': [''],  # hardcoded on purpose for subtitle analysis TODO: fix this
    "something I didn't expected to test with l'huillier.": ['didnt', 'expected', 'test', 'lhuillier'],
    "l'huillier is a last name a will not change.": ["l'huillier", "change"],
    "didn't will be removed (stopword).": ["removed", 'stopword'],
    '': ['']
}
TERMS_MAPPING = {'dog': 'cat'}
TEST_CASES_TERMS_MAPPING = {'this is a dog': 'this is a cat'}


class TestUtilText(unittest.TestCase):
    def test_process_content(self):
        for input_test, expected_output in TEST_CASES.items():
            output = process_content(input_test)
            self.assertEqual(output, expected_output)

    def test_process_content_with_terms_mapping(self):
        for input_test, expected_output in TEST_CASES.items():
            output = process_content(input_test, terms_mapping=TERMS_MAPPING)
            self.assertEqual(output, expected_output)
