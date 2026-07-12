import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.exception import Custom_Exception, error_message_details


class TestCustomException(unittest.TestCase):
    def test_error_message_details_includes_file_line_and_message(self):
        try:
            1 / 0
        except Exception as e:
            message = error_message_details(e, error_detail=sys)

        self.assertIn(__file__, message)
        self.assertIn("division by zero", message)
        self.assertIn("line number", message)

    def test_custom_exception_str_matches_formatted_message(self):
        try:
            1 / 0
        except Exception as e:
            expected = error_message_details(e, error_detail=sys)
            custom_exception = Custom_Exception(e, error_details=sys)

        self.assertEqual(str(custom_exception), expected)

    def test_custom_exception_is_an_exception(self):
        try:
            1 / 0
        except Exception as e:
            custom_exception = Custom_Exception(e, error_details=sys)

        self.assertIsInstance(custom_exception, Exception)


if __name__ == "__main__":
    unittest.main()
