from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from model import load_obb_checkpoint
from pipeline import OBBInferencePipeline
from postprocess import xywhr_to_corners
from preprocess import discover_images, letterbox


ROOT = Path(__file__).resolve().parent


def run_configured_input_test() -> list[dict]:
    """Run every image under config.test.input and write config.test.output."""

    pipeline = OBBInferencePipeline(ROOT / "config" / "config.yaml")
    return pipeline.run_test()


class StandaloneOBBPipelineTests(unittest.TestCase):
    def test_checkpoint_loads_without_ultralytics_model_classes(self) -> None:
        model = load_obb_checkpoint(ROOT / "weight" / "best.pt", torch.device("cpu"))
        self.assertEqual(model.names[0], "watermark")
        self.assertEqual(model.model[-1].nc, 1)

    def test_standalone_checkpoint_contains_no_ultralytics_reference(self) -> None:
        path = ROOT / "weight" / "best_standalone.pt"
        model = load_obb_checkpoint(path, torch.device("cpu"))
        self.assertIsInstance(model, torch.jit.ScriptModule)
        self.assertNotIn(b"ultralytics", path.read_bytes().lower())

    def test_letterbox_preserves_aspect_ratio_for_large_and_small_images(self) -> None:
        for shape in ((3000, 500, 3), (40, 120, 3)):
            image = np.zeros(shape, dtype=np.uint8)
            output, info = letterbox(image, (960, 960), (114, 114, 114), True)
            self.assertEqual(output.shape, (960, 960, 3))
            self.assertAlmostEqual(info.scale, min(960 / shape[1], 960 / shape[0]))

    def test_xywhr_conversion_returns_four_corners(self) -> None:
        corners = xywhr_to_corners(torch.tensor([50.0, 40.0, 20.0, 10.0, 0.0]))
        self.assertEqual(tuple(corners.shape), (4, 2))

    def test_configured_input_folder_inference(self) -> None:
        results = run_configured_input_test()
        pipeline = OBBInferencePipeline(ROOT / "config" / "config.yaml")
        input_images = discover_images(pipeline.test_input, pipeline.test_recursive)

        self.assertEqual(len(results), len(input_images))
        self.assertGreater(len(results), 0)
        for result in results:
            self.assertIn("outputs", result)
            self.assertTrue(Path(result["outputs"]["json"]).is_file())
            self.assertTrue(Path(result["outputs"]["annotated_image"]).is_file())
            self.assertEqual(result["annotation_type"], "obb_prediction")
            for watermark in result["watermarks"]:
                self.assertEqual(len(watermark["obb_label"]), 9)


if __name__ == "__main__":
    unittest.main()
