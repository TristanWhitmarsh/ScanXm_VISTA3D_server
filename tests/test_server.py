import importlib
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


server = importlib.import_module("ScanXm_VISTA3D_server")


class ServerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary_uploads = tempfile.TemporaryDirectory()
        server.UPLOAD_DIR = Path(self.temporary_uploads.name)
        server._CURRENT_SESSION_ID = None
        server._CANCEL_EVENT.clear()
        with server._JOB_LOCK:
            server._JOB.update(
                {
                    "generation": 0,
                    "status": "idle",
                    "message": "",
                    "payload": None,
                    "headers": {},
                }
            )
        self.client = server.app.test_client()

    def tearDown(self):
        self.temporary_uploads.cleanup()

    def url(self, path):
        return f"/{server.SERVER_KEY}{path}"

    def test_generated_key_is_not_a_short_example_key(self):
        self.assertGreaterEqual(len(server.SERVER_KEY), 32)
        self.assertNotEqual(server.SERVER_KEY, "1234")

    def test_model_directory_is_fixed_beside_server_files(self):
        expected = Path(server.nv_segment_worker.__file__).resolve().parent / "NV-Segment-CT"
        self.assertEqual(server.nv_segment_worker.MODEL_DIR, expected)

    def test_wrong_key_is_rejected(self):
        response = self.client.get("/wrong/info/")
        self.assertEqual(response.status_code, 401)

    def test_info_advertises_only_commercial_ct_models(self):
        response = self.client.get(
            self.url("/info/"), headers={"X-Session-ID": "client-a"}
        )
        self.assertEqual(response.status_code, 200)
        names = {
            item["label"]
            for group in response.get_json()["groups"]
            for item in group["items"]
        }
        self.assertEqual(names, {"CT_Full", "CT_Interactive"})
        self.assertNotIn("MR_Full", names)

    def test_new_scanxm_session_cleans_previous_state(self):
        with mock.patch.object(server.nv_segment_worker, "stop_nv_all") as stop:
            first = self.client.get(
                self.url("/info/"), headers={"X-Session-ID": "client-a"}
            )
            second = self.client.get(
                self.url("/info/"), headers={"X-Session-ID": "client-b"}
            )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        stop.assert_called_once()
        self.assertEqual(server._CURRENT_SESSION_ID, "client-b")

    def test_upload_id_cannot_escape_temporary_directory(self):
        response = self.client.post(
            self.url("/upload_chunk?upload_id=../outside"),
            data=b"1234",
            content_type="application/octet-stream",
            headers={"X-Session-ID": "client-a"},
        )
        self.assertEqual(response.status_code, 400)

    def test_chunked_full_volume_job_preserves_scanxm_wire_format(self):
        headers = {"X-Session-ID": "client-a"}
        upload = self.client.post(
            self.url("/upload_chunk?upload_id=volume-1"),
            data=b"\x00\x00\x00\x00",
            content_type="application/octet-stream",
            headers=headers,
        )
        self.assertEqual(upload.status_code, 200)

        result_headers = {
            "X-DType": "uint16",
            "X-Order": "DHW",
            "X-Width": "1",
            "X-Height": "1",
            "X-Depth": "1",
            "X-Model": "CT_Full",
        }
        with mock.patch.object(
            server,
            "run_nv_segment_full",
            return_value=(b"\x01\x00", result_headers),
        ):
            response = self.client.post(
                self.url("/initmodel"),
                data={
                    "model": "CT_Full",
                    "upload_id": "volume-1",
                    "width": "1",
                    "height": "1",
                    "depth": "1",
                    "dtype": "float32",
                    "spacing": "1,1,1",
                    "origin": "0,0,0",
                },
                headers=headers,
            )
            self.assertEqual(response.status_code, 202)

            result = None
            for _ in range(100):
                result = self.client.get(self.url("/getresult"), headers=headers)
                if result.status_code != 204:
                    break
                time.sleep(0.01)

        self.assertIsNotNone(result)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.data, b"\x01\x00")
        self.assertEqual(result.headers["X-DType"], "uint16")
        self.assertEqual(result.headers["X-Model"], "CT_Full")

    def test_interactive_initialization_returns_scanxm_capabilities(self):
        headers = {"X-Session-ID": "client-a"}
        self.client.post(
            self.url("/upload_chunk?upload_id=interactive-1"),
            data=b"\x00\x00\x00\x00",
            content_type="application/octet-stream",
            headers=headers,
        )
        capabilities = {
            "region": False,
            "point": True,
            "mask": False,
            "apply_4d": False,
            "reset": True,
            "interactive_3d": True,
        }
        with mock.patch.object(
            server.nv_segment_worker,
            "init_nv_interactive",
            return_value=capabilities,
        ):
            response = self.client.post(
                self.url("/initmodel"),
                data={
                    "model": "CT_Interactive",
                    "upload_id": "interactive-1",
                    "width": "1",
                    "height": "1",
                    "depth": "1",
                    "dtype": "float32",
                    "spacing": "1,1,1",
                    "origin": "0,0,0",
                },
                headers=headers,
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["capabilities"], capabilities)


if __name__ == "__main__":
    unittest.main()
