import unittest
import numpy as np
from fume_bounds.evidence_feedback.student import (
    AdamStudent, design, embedding, initialize, loss_and_grad,
)


class StudentTests(unittest.TestCase):
    def test_gradient_by_central_difference(self):
        rng = np.random.default_rng(517)
        image, text = design(rng.normal(size=(7, 2))), design(rng.normal(size=(7, 2)))
        labels = np.array([0, 1, 1, 0, 1, 0, 0])
        parameters = initialize(111)
        _, gradient, _ = loss_and_grad(parameters, image, text, labels)
        for name, weights in parameters.items():
            for index in np.ndindex(weights.shape):
                original = weights[index]
                weights[index] = original + 1e-6
                high = loss_and_grad(parameters, image, text, labels)[0]
                weights[index] = original - 1e-6
                low = loss_and_grad(parameters, image, text, labels)[0]
                weights[index] = original
                self.assertAlmostEqual((high-low)/2e-6, gradient[name][index], delta=2e-7)

    def test_loss_against_literal_archived_formula(self):
        # Independent forward translation of processor.calc_loss, no gradients.
        rng = np.random.default_rng(12)
        image, text = design(rng.normal(size=(9, 2))), design(rng.normal(size=(9, 2)))
        y = np.eye(2)[rng.integers(0, 2, size=9)]
        p = initialize(44)
        views = [embedding(image, p["image"]), embedding(text, p["text"])]
        loss_fuzzy = 0.
        for view in views:
            pred = np.maximum(view, 0.)
            wrong = (pred*(1-y)).max(1, keepdims=True)
            correct = (pred*y).max(1, keepdims=True)
            necessity = (1-correct)*(1-y) + (1-wrong)*y
            credibility = (pred+necessity)/2
            loss_fuzzy += np.linalg.norm(credibility-y, axis=1).mean()
        all_features = np.concatenate(views)
        sim = np.exp(all_features @ all_features.T)
        sim -= np.diag(np.diag(sim))
        sum1 = sim[:, :9] + sim[:, 9:]
        diag1 = np.r_[np.diag(sum1[:9]), np.diag(sum1[9:])]
        sum2 = sim[:9] + sim[9:]
        diag2 = np.r_[np.diag(sum2[:, :9]), np.diag(sum2[:, 9:])]
        loss_cl = -np.log(diag1/sim.sum(1)).mean()-np.log(diag2/sim.sum(1)).mean()
        value = loss_and_grad(p, image, text, y.argmax(1), weight_decay=0.)[0]
        self.assertAlmostEqual(value, loss_fuzzy+.5*loss_cl, delta=1e-12)

    def test_every_entity_and_continuous_adam(self):
        rng = np.random.default_rng(1)
        image, text = design(rng.normal(size=(65, 2))), design(rng.normal(size=(65, 2)))
        labels = rng.integers(0, 2, 65)
        model = AdamStudent(initialize(3))
        for expected in (1, 2):
            record = model.train_epoch(image, text, labels, 64, rng)
            self.assertEqual(record["entities"], 65)
            self.assertEqual(record["batches"], 1)
            self.assertEqual(record["cumulative_adam_steps"], expected)


if __name__ == "__main__":
    unittest.main()
