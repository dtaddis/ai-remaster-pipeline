from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from model_paths import huggingface_cache_dir, resolve_comfy_model_path  # noqa: E402
import dependency_manager  # noqa: E402


class HuggingFaceCacheTests(unittest.TestCase):
    def test_hf_home_uses_standard_hub_subdirectory(self) -> None:
        with mock.patch.dict(os.environ, {"HF_HOME": r"D:\shared\huggingface"}, clear=True):
            self.assertEqual(huggingface_cache_dir(Path("fallback")), Path(r"D:\shared\huggingface") / "hub")

    def test_explicit_hub_cache_wins_over_hf_home(self) -> None:
        env = {"HF_HOME": r"D:\shared\huggingface", "HF_HUB_CACHE": r"E:\model-cache"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(huggingface_cache_dir(Path("fallback")), Path(r"E:\model-cache"))

    def test_arp_override_has_highest_priority(self) -> None:
        env = {
            "HF_HOME": r"D:\shared\huggingface",
            "HF_HUB_CACHE": r"E:\model-cache",
            "ARP_HF_CACHE_DIR": r"F:\arp-cache",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(huggingface_cache_dir(Path("fallback")), Path(r"F:\arp-cache"))


class ComfyModelPathTests(unittest.TestCase):
    def test_default_download_model_base_receives_model_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            comfy = root / "ComfyUI"
            comfy.mkdir()
            (comfy / "extra_model_paths.yaml").write_text(
                """
shared:
  base_path: ../shared
  is_default: true
  download_model_base: models
  checkpoints: models/checkpoints
  diffusion_models: |
    models/diffusion_models
    models/unet
""",
                encoding="utf-8",
            )

            resolved = resolve_comfy_model_path(comfy, "models/unet/model.gguf")

            self.assertEqual(resolved, root / "shared" / "models" / "unet" / "model.gguf")

    def test_default_category_path_is_used_without_download_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            comfy = root / "ComfyUI"
            comfy.mkdir()
            (comfy / "extra_model_paths.yaml").write_text(
                """
shared:
  base_path: ../shared
  is_default: true
  loras: weights/loras
""",
                encoding="utf-8",
            )

            resolved = resolve_comfy_model_path(comfy, "models/loras/fix.safetensors")

            self.assertEqual(resolved, root / "shared" / "weights" / "loras" / "fix.safetensors")

    def test_existing_model_in_nondefault_search_path_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            comfy = root / "ComfyUI"
            comfy.mkdir()
            existing = root / "archive" / "text" / "encoder.safetensors"
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"model")
            (comfy / "extra_model_paths.yaml").write_text(
                """
archive:
  base_path: ../archive
  text_encoders: text
""",
                encoding="utf-8",
            )

            resolved = resolve_comfy_model_path(comfy, "models/text_encoders/encoder.safetensors")

            self.assertEqual(resolved, existing)

    def test_missing_config_falls_back_to_comfy_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            comfy = Path(tmp_text) / "ComfyUI"

            resolved = resolve_comfy_model_path(comfy, "models/vae/model.safetensors")

            self.assertEqual(resolved, comfy / "models" / "vae" / "model.safetensors")


class DependencyManagerPathTests(unittest.TestCase):
    def test_outpaint_downloads_only_official_video_lora(self) -> None:
        with mock.patch.object(dependency_manager, "ensure_hf_models") as ensure:
            dependency_manager.ensure_outpaint_models(Path("ComfyUI"))

        models = ensure.call_args.args[1]
        loras = [model for model in models if model.destination.startswith("models/loras/")]
        self.assertEqual(
            [model.file for model in loras],
            [dependency_manager.DEFAULT_OUTPAINT_LORA],
        )
        self.assertFalse(any("audio_vae" in model.file for model in models))

    def test_outpaint_can_select_oumoumad_video_lora(self) -> None:
        with mock.patch.object(dependency_manager, "ensure_hf_models") as ensure:
            dependency_manager.ensure_outpaint_models(
                Path("ComfyUI"),
                outpaint_lora=dependency_manager.OUMOUMAD_OUTPAINT_LORA,
            )

        models = ensure.call_args.args[1]
        loras = [model for model in models if model.destination.startswith("models/loras/")]
        self.assertEqual([model.file for model in loras], [dependency_manager.OUMOUMAD_OUTPAINT_LORA])

    def test_ltx25_outpaint_uses_quantized_models_and_existing_official_lora(self) -> None:
        with mock.patch.object(dependency_manager, "ensure_hf_models") as ensure:
            dependency_manager.ensure_ltx25_outpaint_models(Path("ComfyUI"))

        models = ensure.call_args.args[1]
        destinations = {model.destination for model in models}
        self.assertIn(f"models/diffusion_models/{dependency_manager.LTX25_GGUF_MODEL}", destinations)
        self.assertIn(f"models/text_encoders/{dependency_manager.LTX25_TEXT_ENCODER}", destinations)
        self.assertIn(f"models/latent_upscale_models/{dependency_manager.LTX25_LATENT_UPSCALER}", destinations)
        self.assertIn(f"models/loras/{dependency_manager.DEFAULT_OUTPAINT_LORA}", destinations)

    def test_gated_model_403_is_rewritten_as_actionable_access_error(self) -> None:
        class Response:
            status_code = 403

        denied = RuntimeError("download denied")
        denied.response = Response()
        with tempfile.TemporaryDirectory() as tmp_text:
            comfy = Path(tmp_text) / "ComfyUI"
            comfy.mkdir()
            model = dependency_manager.HfModel("owner/gated-model", "model.safetensors", "models/loras/model.safetensors")
            with (
                mock.patch.object(dependency_manager, "ensure_huggingface_hub"),
                mock.patch.object(dependency_manager, "remote_file_size", return_value=0),
                mock.patch.object(dependency_manager, "download_hf_file", side_effect=denied),
                self.assertRaises(dependency_manager.HuggingFaceAccessError) as raised,
            ):
                dependency_manager.ensure_hf_models(comfy, [model])

        self.assertIn("https://huggingface.co/owner/gated-model", str(raised.exception))
        self.assertIn("approve access", str(raised.exception))

    def test_download_uses_configured_cache_and_comfy_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            comfy = root / "ComfyUI"
            comfy.mkdir()
            (comfy / "extra_model_paths.yaml").write_text(
                """
shared:
  base_path: ../shared
  is_default: true
  download_model_base: models
""",
                encoding="utf-8",
            )
            downloaded = root / "downloaded.safetensors"
            downloaded.write_bytes(b"model")
            model = dependency_manager.HfModel("owner/repo", "model.safetensors", "models/vae/model.safetensors")

            with (
                mock.patch.dict(os.environ, {"HF_HOME": str(root / "hf-home")}, clear=True),
                mock.patch.object(dependency_manager, "ensure_huggingface_hub"),
                mock.patch.object(dependency_manager, "remote_file_size", return_value=5),
                mock.patch.object(dependency_manager, "download_hf_file", return_value=downloaded) as download,
                mock.patch.object(dependency_manager, "copy_model_file") as copy,
            ):
                dependency_manager.ensure_hf_models(comfy, [model])

            download.assert_called_once_with("owner/repo", "model.safetensors", root / "hf-home" / "hub", 5)
            copy.assert_called_once_with(
                downloaded,
                root / "shared" / "models" / "vae" / "model.safetensors",
                5,
            )


if __name__ == "__main__":
    unittest.main()
