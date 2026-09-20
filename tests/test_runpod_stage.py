from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import runpod_stage  # noqa: E402


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return b'{"id":"pod-1"}'


class RunPodAPIRequestTests(unittest.TestCase):
    def test_custom_outpaint_mask_is_uploaded_and_rewritten(self) -> None:
        with tempfile.TemporaryDirectory(dir=runpod_stage.ROOT) as folder_text:
            mask = Path(folder_text) / "mask.png"
            mask.write_bytes(b"mask")
            command = [sys.executable, "-u", str(runpod_stage.ROOT / "scripts" / "outpaint_video.py"), "--custom-mask", str(mask)]
            converted = runpod_stage.remote_command(command)

        self.assertEqual(converted[converted.index("--custom-mask") + 1], runpod_stage.remote_path(str(mask)))

    def test_bootstrap_reuses_matching_cuda_13_image_runtime(self) -> None:
        bootstrap = (runpod_stage.ROOT / "scripts" / "bootstrap_runpod.sh").read_text(encoding="utf-8")

        self.assertIn("python3 -m venv --system-site-packages", bootstrap)
        self.assertIn("torchvision==0.24.1+cu130", bootstrap)
        self.assertIn("torchaudio==2.9.1+cu130", bootstrap)

    def test_create_pod_uses_live_availability_priority_and_network_volume(self) -> None:
        args = mock.Mock(
            image="cuda13", gpu_types="gpu-a,gpu-b", volume_gb=100,
            network_volume_id="volume-1", api_key="secret",
        )
        with mock.patch.object(runpod_stage, "api_request", return_value={"id": "pod-1"}) as request:
            runpod_stage.create_pod(args, "ssh-public")

        payload = request.call_args.args[3]
        self.assertEqual(payload["gpuTypePriority"], "availability")
        self.assertEqual(payload["networkVolumeId"], "volume-1")
        self.assertNotIn("volumeInGb", payload)

    def test_automatic_network_volume_creation_is_remembered(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = Path(folder) / "settings.json"
            settings.write_text("{}", encoding="utf-8")
            args = mock.Mock(
                storage_mode="network", network_volume_id="", volume_gb=100,
                data_center_id="EU-RO-1", api_key="secret", settings_file=str(settings),
            )
            with mock.patch.object(runpod_stage, "api_request", return_value={"id": "volume-1"}) as request:
                runpod_stage.ensure_network_volume(args)

            self.assertEqual(args.network_volume_id, "volume-1")
            stored = json.loads(settings.read_text(encoding="utf-8"))
            self.assertEqual(stored["cloud"]["runpod_network_volume_id"], "volume-1")
            self.assertEqual(request.call_args.args[2], "/networkvolumes")

    def test_saved_host_tied_pod_is_replaced_when_portable_storage_is_selected(self) -> None:
        args = mock.Mock(pod_id="old-pod", network_volume_id="volume-1", api_key="secret")
        old_pod = {"id": "old-pod", "desiredStatus": "STOPPED", "networkVolume": None}
        new_pod = {"id": "new-pod", "publicIp": "127.0.0.1", "portMappings": {"22": 2200}}
        with (
            mock.patch.object(runpod_stage, "api_request", return_value=old_pod),
            mock.patch.object(runpod_stage, "create_pod", return_value=new_pod) as create,
        ):
            pod, host, port = runpod_stage.wait_for_pod(args, "ssh-public")

        self.assertEqual(pod["id"], "new-pod")
        self.assertEqual((host, port), ("127.0.0.1", 2200))
        self.assertEqual(args.pod_id, "")
        create.assert_called_once_with(args, "ssh-public")

    def test_saved_pod_reuses_rest_network_volume_id_shape(self) -> None:
        args = mock.Mock(pod_id="saved-pod", network_volume_id="volume-1", api_key="secret")
        saved_pod = {
            "id": "saved-pod",
            "desiredStatus": "RUNNING",
            "networkVolumeId": "volume-1",
            "publicIp": "127.0.0.1",
            "portMappings": {"22": 2200},
        }
        with (
            mock.patch.object(runpod_stage, "api_request", return_value=saved_pod),
            mock.patch.object(runpod_stage, "create_pod") as create,
        ):
            pod, host, port = runpod_stage.wait_for_pod(args, "ssh-public")

        self.assertEqual(pod["id"], "saved-pod")
        self.assertEqual((host, port), ("127.0.0.1", 2200))
        create.assert_not_called()

    def test_credentials_load_from_machine_local_settings(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = Path(folder) / "settings.json"
            settings.write_text(
                json.dumps({"cloud": {
                    "runpod_api_key": "rp-local",
                    "huggingface_token": "hf-local",
                    "runpod_pod_id": "pod-local",
                    "runpod_network_volume_id": "volume-local",
                }}),
                encoding="utf-8",
            )
            args = mock.Mock(
                api_key="", huggingface_token="", pod_id="", network_volume_id="",
                settings_file=str(settings),
            )

            runpod_stage.load_local_cloud_secrets(args)

        self.assertEqual(args.api_key, "rp-local")
        self.assertEqual(args.huggingface_token, "hf-local")
        self.assertEqual(args.pod_id, "pod-local")
        self.assertEqual(args.network_volume_id, "volume-local")

    def test_get_retries_transient_timeout(self) -> None:
        with (
            mock.patch.object(
                runpod_stage.urllib.request,
                "urlopen",
                side_effect=[TimeoutError("temporary"), _Response()],
            ) as urlopen,
            mock.patch.object(runpod_stage.time, "sleep") as sleep,
        ):
            result = runpod_stage.api_request("secret", "GET", "/pods/pod-1")

        self.assertEqual(result, {"id": "pod-1"})
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(2)

    def test_post_is_not_retried_to_avoid_duplicate_creation(self) -> None:
        with (
            mock.patch.object(
                runpod_stage.urllib.request,
                "urlopen",
                side_effect=TimeoutError("temporary"),
            ) as urlopen,
            self.assertRaisesRegex(RuntimeError, "temporarily unreachable"),
        ):
            runpod_stage.api_request("secret", "POST", "/pods", {"name": "test"})

        self.assertEqual(urlopen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
