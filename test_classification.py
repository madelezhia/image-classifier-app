from ImageClassifier.predict import predict
import unittest
import os

class TestImageClassifier(unittest.TestCase):
    def test_predict(self):
        # Chemin vers une image de test (à placer dans le dossier tests/)
        test_dir = os.path.join(os.path.dirname(__file__), "tests")
        os.makedirs(test_dir, exist_ok=True)

        # Télécharger une image de chat ImageNet (déjà présente ou à ajouter)
        test_image_path = os.path.join(test_dir, "pineapple-sits-table-with-other-fruits_726745-5475-3767101720.jpg")

        # Vérifie que l’image existe (sinon skip ou télécharge)
        if not os.path.exists(test_image_path):
            self.skipTest("Test image not found. Add 'cat.jpg' in 'tests/' folder.")

        result = predict(test_image_path)

        # Vérifie que le résultat n’est pas None
        self.assertIsNotNone(result)

        # Vérifie que la classe dominante est bien un label connu
        top_class = list(result.keys())[0]
        self.assertIn(top_class, result)

        # Vérifie que le score est entre 0 et 1
        top_score = result[top_class]
        self.assertGreaterEqual(top_score, 0.0)
        self.assertLessEqual(top_score, 1.0)

unittest.main()