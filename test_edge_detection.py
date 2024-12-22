import unittest
import numpy as np
from parallel_edge_detection import DistributedEdgeDetector, ValidationError

class TestEdgeDetection(unittest.TestCase):
    def setUp(self):
        print("\nSetting up test...")
        self.detector = DistributedEdgeDetector()
        self.test_image = np.random.rand(1024, 1024)

    def test_validation(self):
        print("Testing operator validation...")
        with self.assertRaises(ValidationError):
            self.detector.validate_operator('invalid_operator')
        print("✓ Operator validation test passed")

    def test_preprocessing(self):
        print("Testing image preprocessing...")
        # Create a test image file
        from PIL import Image
        test_img = Image.fromarray((self.test_image * 255).astype(np.uint8))
        test_img.save("test_data/valid.jpg")
        
        processed = self.detector.preprocess_image("test_data/valid.jpg")
        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape, (1024, 1024))
        print("✓ Preprocessing test passed")

    def test_partitioning(self):
        print("Testing image partitioning...")
        partitions = self.detector.create_overlapping_partitions(
            self.test_image, 4, overlap=1
        )
        self.assertEqual(len(partitions), 4)
        print("✓ Partitioning test passed")

    def test_operator_application(self):
        print("Testing operator application...")
        result = self.detector.apply_operator(
            self.test_image, 
            self.detector.operators.sobel()['x']
        )
        self.assertIsNotNone(result)
        print("✓ Operator application test passed")

def run_tests():
    # Create test directory if it doesn't exist
    import os
    if not os.path.exists("test_data"):
        os.makedirs("test_data")
    
    # Run tests with verbosity
    print("Starting Edge Detection Tests...")
    unittest.main(verbosity=2)

if __name__ == "__main__":
    run_tests() 