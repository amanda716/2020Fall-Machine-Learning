import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import qwen_call


class TestQwenCall(unittest.TestCase):
    @patch("qwen_call.OpenAI")
    def test_call_qwen_happy_path(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

        result = qwen_call.call_qwen(
            prompt="你好",
            api_key="test-key",
            model="qwen-plus",
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            temperature=0.3,
        )

        self.assertEqual(result, "ok")
        mock_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        mock_client.chat.completions.create.assert_called_once()

    def test_resolve_api_key_from_cli(self):
        self.assertEqual(qwen_call._resolve_api_key("cli-key"), "cli-key")

    @patch("qwen_call.os.getenv", return_value=None)
    def test_resolve_api_key_missing(self, _mock_getenv):
        with self.assertRaises(ValueError):
            qwen_call._resolve_api_key(None)


if __name__ == "__main__":
    unittest.main()
