import tempfile
import unittest
from pathlib import Path

from scripts.common import Match, threshold_probability
from scripts.predict_results import optimize
from scripts.train_model import train


class PipelineTests(unittest.TestCase):
    def test_tie_priority_is_one_then_two_then_x(self):
        game = Match(1, 1, "A", "B", {"1": .4, "X": .2, "2": .4})
        self.assertEqual(game.ranking, ["1", "2", "X"])

    def test_ticket_shape_and_flamengo_constraint(self):
        games = [Match(1, i + 1, "A", "B", {"1": .5, "X": .3, "2": .2}) for i in range(14)]
        games[13] = Match(1, 14, "A", "FLAMENGO/RJ", {"1": .8, "X": .15, "2": .05})
        ticket, _ = optimize(games, "uncertainty")
        self.assertEqual(sorted(map(len, ticket)), [1] * 9 + [2] * 5)
        self.assertIn("2", ticket[13])
        self.assertEqual(sum(map(len, ticket)), 19)

    def test_threshold_probability_extremes(self):
        self.assertEqual(threshold_probability([1.0] * 14), 1.0)
        self.assertEqual(threshold_probability([0.0] * 14), 0.0)

    def test_training_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            first = train("data/concursos_anteriores.csv", str(path))
            second = train("data/concursos_anteriores.csv", str(path))
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
