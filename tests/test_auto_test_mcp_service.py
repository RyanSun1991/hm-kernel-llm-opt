import unittest
from pathlib import Path
from unittest import mock

from hmopt.api import auto_test_mcp_service as svc


class AutoTestMcpServiceTests(unittest.TestCase):
    def test_builtin_swipe_uses_env_target_without_connect(self) -> None:
        run_calls = []

        def fake_run(argv, timeout_s):
            run_calls.append(argv)
            return {
                "command": " ".join(argv),
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "duration_s": 0.0,
            }

        with mock.patch.object(svc, "_run_cmd", side_effect=fake_run), mock.patch.object(
            svc, "_resolve_hdc_bin", return_value="hdc"
        ), mock.patch.object(svc, "_resolve_local_result_dir", return_value=Path("outputs/test")), mock.patch.object(
            svc, "_builtin_swipe_local_script", return_value=Path("scripts/phone_tests/basic_swipe.sh")
        ), mock.patch.object(svc, "_resolve_default_target", return_value="2LQ0223A31014882"):
            result = svc.run_phone_test(
                target="",
                test_case=svc.BUILTIN_SWIPE_CASE,
                remote_script="",
                remote_result_path="",
                connect_before_shell=False,
                use_target_flag=True,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["target"], "2LQ0223A31014882")
        self.assertEqual(result["steps"]["connect"], None)
        self.assertEqual(run_calls[0][0:5], ["hdc", "-t", "2LQ0223A31014882", "file", "send"])
        self.assertEqual(run_calls[1][0:4], ["hdc", "-t", "2LQ0223A31014882", "shell"])
        self.assertEqual(run_calls[2][0:4], ["hdc", "-t", "2LQ0223A31014882", "shell"])
        self.assertEqual(run_calls[3][0:5], ["hdc", "-t", "2LQ0223A31014882", "file", "recv"])

    def test_use_target_flag_requires_target_or_env(self) -> None:
        with self.assertRaisesRegex(ValueError, "HMOPT_AUTO_TEST_TARGET"):
            svc.run_phone_test(
                target="",
                test_case="boot_smoke",
                remote_script="/data/local/tmp/run_test.sh",
                remote_result_path="/data/local/tmp/boot.result",
                connect_before_shell=False,
                use_target_flag=True,
            )

    def test_regular_case_requires_remote_script(self) -> None:
        with self.assertRaisesRegex(ValueError, "remote_script is required"):
            svc._build_shell_command(
                remote_script="",
                test_case="boot_smoke",
                extra_args=[],
            )

    def test_builtin_swipe_extra_args_resolution(self) -> None:
        self.assertEqual(
            svc._resolve_builtin_swipe_args("/data/local/tmp/a.result", []),
            ["/data/local/tmp/a.result", "60", "1050"],
        )
        self.assertEqual(
            svc._resolve_builtin_swipe_args("/data/local/tmp/a.result", ["33", "9"]),
            ["/data/local/tmp/a.result", "33", "9"],
        )

    def test_cli_main_success_exit_code_zero(self) -> None:
        fake = {"success": True, "test_case": "basic_swipe"}
        with mock.patch.object(svc, "run_phone_test", return_value=fake):
            code = svc._main(["--test-case", "basic_swipe", "--remote-script", "/data/local/tmp/x.sh"])
        self.assertEqual(code, 0)

    def test_cli_main_error_exit_code_one(self) -> None:
        with mock.patch.object(svc, "run_phone_test", side_effect=ValueError("boom")):
            code = svc._main(["--test-case", "basic_swipe", "--remote-script", "/data/local/tmp/x.sh"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
