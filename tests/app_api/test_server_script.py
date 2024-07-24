import unittest
from unittest.mock import patch
from tracr.app_api.server_script import SERVER_SCRIPT


class TestServerScript(unittest.TestCase):

    @patch(
        "builtins.open", new_callable=unittest.mock.mock_open, read_data=SERVER_SCRIPT
    )
    def test_server_script_reading(self, mock_open):
        with open("dummy_path", "r") as f:
            script_content = f.read()
        self.assertIn("import sys", script_content)
