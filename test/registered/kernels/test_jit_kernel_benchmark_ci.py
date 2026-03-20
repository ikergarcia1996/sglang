import os
import subprocess
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=900, suite="stage-b-kernel")


class TestJITKernelBenchmarkCI(unittest.TestCase):
    def test_jit_kernel_benchmarks(self):
        repo_root = Path(__file__).resolve().parents[3]
        benchmark_dir = repo_root / "python" / "sglang" / "jit_kernel" / "benchmark"
        env = os.environ.copy()
        env["CI"] = "true"

        failures = []
        for bench_file in sorted(benchmark_dir.glob("bench_*.py")):
            cmd = [sys.executable, str(bench_file)]
            try:
                result = subprocess.run(
                    cmd,
                    cwd=benchmark_dir,
                    env=env,
                    timeout=120,
                )
            except subprocess.TimeoutExpired:
                failures.append(f"{bench_file.name} (timeout)")
                continue

            if result.returncode != 0:
                failures.append(f"{bench_file.name} (exit code {result.returncode})")

        self.assertFalse(
            failures, "jit-kernel benchmark tests failed: " + ", ".join(failures)
        )


if __name__ == "__main__":
    unittest.main()
