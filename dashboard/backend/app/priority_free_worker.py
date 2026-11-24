"""Lazy, serialized manager for the persistent priority-free DQN worker."""

import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable

from .config import REPOSITORY_ROOT, Settings, get_settings

_WORKER_EOF = object()


class WorkerManagerError(RuntimeError):
    """Base class for worker startup, protocol, and runtime failures."""


class WorkerStartupError(WorkerManagerError):
    """The Python worker could not be started."""


class WorkerUnavailableError(WorkerManagerError):
    """The worker exited or its input/output pipe failed."""


class WorkerTimeoutError(WorkerManagerError):
    """The worker did not answer before the configured timeout."""


class WorkerProtocolError(WorkerManagerError):
    """The worker returned invalid or unexpected JSON."""


class PriorityFreeDqnWorkerManager:
    """Own one on-demand Python process and serialize its request/response pairs."""

    def __init__(
        self,
        *,
        settings_provider: Callable[[], Settings] = get_settings,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self._settings_provider = settings_provider
        self._popen_factory = popen_factory
        self._lock = threading.Lock()
        self._process = None
        self._reader_thread = None
        self._responses: queue.Queue = queue.Queue()
        self._shutdown = False

    def _environment(self, settings: Settings) -> dict[str, str]:
        environment = os.environ.copy()
        runtime_dirs = [
            settings.priority_free_dqn_cpp_engine_path.parent,
            settings.drl_python_path.parent,
        ]
        configured_runtime = os.getenv("MSYS2_UCRT64_BIN")
        runtime_dirs.append(
            Path(configured_runtime) if configured_runtime else Path(r"C:\msys64\ucrt64\bin")
        )
        current_path = environment.get("PATH", "")
        existing = {entry.casefold() for entry in current_path.split(os.pathsep) if entry}
        additions = [str(path) for path in runtime_dirs
                     if path.is_dir() and str(path).casefold() not in existing]
        environment["PATH"] = os.pathsep.join(additions + ([current_path] if current_path else []))
        return environment

    @staticmethod
    def _read_stdout(process, responses: queue.Queue) -> None:
        try:
            for line in iter(process.stdout.readline, ""):
                responses.put(line)
        finally:
            responses.put(_WORKER_EOF)

    def _stop_locked(self) -> None:
        process = self._process
        reader = self._reader_thread
        self._process = None
        self._reader_thread = None
        self._responses = queue.Queue()
        if process is None:
            return

        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                    process.wait(timeout=1)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        for pipe in (getattr(process, "stdin", None), getattr(process, "stdout", None)):
            if pipe is not None:
                try:
                    pipe.close()
                except (OSError, ValueError):
                    pass
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1)

    def _start_locked(self, settings: Settings) -> None:
        command = [
            str(settings.drl_python_path), "-m",
            "drl_scheduler.priority_free_inference_adapter",
            "--worker", "--checkpoint", str(settings.priority_free_dqn_checkpoint_path),
        ]
        try:
            process = self._popen_factory(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                encoding="utf-8",
                bufsize=1,
                cwd=str(REPOSITORY_ROOT),
                env=self._environment(settings),
            )
        except OSError as error:
            raise WorkerStartupError("Could not start the priority-free DQN worker.") from error

        if process.stdin is None or process.stdout is None:
            try:
                process.kill()
            except OSError:
                pass
            raise WorkerStartupError("The priority-free DQN worker pipes could not be opened.")

        responses: queue.Queue = queue.Queue()
        reader = threading.Thread(
            target=self._read_stdout, args=(process, responses),
            name="priority-free-dqn-stdout", daemon=True,
        )
        self._process = process
        self._responses = responses
        self._reader_thread = reader
        reader.start()

    def _ensure_running_locked(self, settings: Settings) -> None:
        if self._shutdown:
            raise WorkerUnavailableError("The priority-free DQN worker manager is shut down.")
        if self._process is not None and self._process.poll() is None:
            return
        if self._process is not None:
            self._stop_locked()
        self._start_locked(settings)

    @staticmethod
    def _validate_response(line: str) -> dict[str, Any]:
        try:
            response = json.loads(line)
        except (json.JSONDecodeError, TypeError) as error:
            raise WorkerProtocolError("The priority-free DQN worker returned invalid JSON.") from error
        if not isinstance(response, dict):
            raise WorkerProtocolError("The priority-free DQN worker response must be a JSON object.")
        if "error" in response:
            error = response.get("error")
            message = error.get("message") if isinstance(error, dict) else None
            raise WorkerProtocolError(
                message or "The priority-free DQN worker rejected the inference request."
            )
        trace = response.get("action_trace")
        completed = response.get("completed_transaction_ids")
        records = response.get("cpp_trace_records")
        if (response.get("algorithm") != "priority_free_variable_dqn"
                or type(response.get("steps")) is not int
                or not isinstance(trace, list)
                or any(type(value) is not int for value in trace)
                or not isinstance(completed, list)
                or any(type(value) is not int for value in completed)
                or not isinstance(records, list)
                or any(not isinstance(value, str) for value in records)
                or response["steps"] != len(trace)):
            raise WorkerProtocolError("The priority-free DQN worker returned an invalid response shape.")
        return response

    def request(self, payload: dict[str, Any], *, settings: Settings | None = None) -> dict[str, Any]:
        """Send one workload and wait for its matching response, under one lock."""
        settings = settings or self._settings_provider()
        with self._lock:
            self._ensure_running_locked(settings)
            process = self._process
            responses = self._responses
            try:
                if process.poll() is not None:
                    raise WorkerUnavailableError("The priority-free DQN worker exited unexpectedly.")
                process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as error:
                self._stop_locked()
                raise WorkerUnavailableError("The priority-free DQN worker input pipe failed.") from error

            try:
                line = responses.get(timeout=settings.simulation_timeout_seconds)
            except queue.Empty as error:
                self._stop_locked()
                raise WorkerTimeoutError("The priority-free DQN worker timed out.") from error
            if line is _WORKER_EOF:
                self._stop_locked()
                raise WorkerUnavailableError("The priority-free DQN worker exited without a response.")
            try:
                return self._validate_response(line)
            except WorkerProtocolError:
                self._stop_locked()
                raise

    def shutdown(self) -> None:
        """Stop the worker if it was started; safe to call repeatedly."""
        with self._lock:
            self._shutdown = True
            self._stop_locked()

    def resume(self) -> None:
        """Allow a later ASGI lifespan to use the manager without starting a process."""
        with self._lock:
            self._shutdown = False


priority_free_dqn_worker = PriorityFreeDqnWorkerManager()
