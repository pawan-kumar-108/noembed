import os
import unittest
from unittest.mock import patch

from src.display import (
    bold,
    contribution_bar,
    dim,
    score_bar,
    score_label,
)


class ColourDisabledTests(unittest.TestCase):
    """
    When stdout is not a TTY (e.g. output is piped/captured, as it always
    is under a test runner), colour must be disabled and every helper
    should return plain text with no escape codes.
    """

    def test_bold_plain_when_not_tty(self):
        with patch("sys.stdout.isatty", return_value=False):
            self.assertEqual(bold("hello"), "hello")

    def test_dim_plain_when_not_tty(self):
        with patch("sys.stdout.isatty", return_value=False):
            self.assertEqual(dim("hello"), "hello")

    def test_score_label_plain_when_not_tty(self):
        with patch("sys.stdout.isatty", return_value=False):
            self.assertEqual(score_label(0.5), "0.5000")

    def test_no_color_env_var_disables_colour_even_on_a_tty(self):
        with patch("sys.stdout.isatty", return_value=True):
            with patch.dict(os.environ, {"NO_COLOR": "1"}):
                self.assertEqual(bold("hello"), "hello")


class ColourEnabledTests(unittest.TestCase):
    def test_bold_wraps_with_escape_codes_when_tty_and_no_color_unset(self):
        with patch("sys.stdout.isatty", return_value=True):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("NO_COLOR", None)
                result = bold("hello")
                self.assertIn("hello", result)
                self.assertIn("\033[", result)  # contains an escape sequence
                self.assertTrue(result.endswith("\033[0m"))  # resets at the end


class ScoreBarTests(unittest.TestCase):
    def test_bar_length_matches_width(self):
        with patch("sys.stdout.isatty", return_value=False):
            bar = score_bar(0.5, width=10)
            self.assertEqual(len(bar), 10)

    def test_zero_score_is_all_empty_blocks(self):
        with patch("sys.stdout.isatty", return_value=False):
            bar = score_bar(0.0, width=10)
            self.assertEqual(bar, "░" * 10)

    def test_full_score_is_all_filled_blocks(self):
        with patch("sys.stdout.isatty", return_value=False):
            bar = score_bar(1.0, width=10)
            self.assertEqual(bar, "█" * 10)

    def test_score_above_one_is_clamped_not_overflowed(self):
        with patch("sys.stdout.isatty", return_value=False):
            bar = score_bar(1.5, width=10)
            self.assertEqual(bar, "█" * 10)

    def test_negative_score_is_clamped_to_zero(self):
        with patch("sys.stdout.isatty", return_value=False):
            bar = score_bar(-0.2, width=10)
            self.assertEqual(bar, "░" * 10)


class ContributionBarTests(unittest.TestCase):
    def test_max_contribution_fills_bar_completely(self):
        with patch("sys.stdout.isatty", return_value=False):
            bar = contribution_bar(0.5, max_contribution=0.5, width=10)
            self.assertEqual(bar, "█" * 10)

    def test_zero_max_contribution_does_not_divide_by_zero(self):
        with patch("sys.stdout.isatty", return_value=False):
            bar = contribution_bar(0.0, max_contribution=0.0, width=10)
            self.assertEqual(bar, "░" * 10)

    def test_half_of_max_fills_roughly_half(self):
        with patch("sys.stdout.isatty", return_value=False):
            bar = contribution_bar(0.25, max_contribution=0.5, width=10)
            self.assertEqual(bar.count("█"), 5)


if __name__ == "__main__":
    unittest.main()