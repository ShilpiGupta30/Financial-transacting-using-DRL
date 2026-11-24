"""Tests for lazy, serialized management of the priority-free DQN worker."""

import json
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app.main import app
from dashboard.backend.app import main as main_module
from dashboard.backend.app.priority_free_worker import (
    PriorityFreeDqnWorkerManager,
    WorkerProtocolError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)


class FakeStdin:
    def __init__(self, process):
        self.process = process
        self.closed = False
        self.pending = ""

    def write(self, value):
        if self.process.mode == "broken_pipe":
            raise BrokenPipeError("simulated broken pipe")
        if self.closed or self.process.poll() is not None:
            raise BrokenPipeError("fake worker stdin is closed")
        self.pending += value
        return len(value)

    def flush(self):
        if self.closed:
            raise BrokenPipeError("fake worker stdin is closed")
        if "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self.process.requests.put(line)

    def close(self):
        if not self.closed:
            self.closed = True
            self.process.requests.put(None)


class FakeStdout:
    def __init__(self, process):
        self.process = process
        self.closed = False

    def readline(self):
        if self.closed:
            return ""
        return self.process.responses.get()

    def close(self):
        self.closed = True
        self.process.responses.put("")


class FakeProcess:
    def __init__(self, mode="normal", *, response_delay=0):
        self.mode = mode
        self.response_delay = response_delay
        self.requests = queue.Queue()
        self.responses = queue.Queue()
        self.returncode = None
        self.stdin = FakeStdin(self)
        self.stdout = FakeStdout(self)
        self.received = []
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        line_number = 0
        while True:
            line = self.requests.get()
            if line is None:
                return
            line_number += 1
            self.received.append(json.loads(line))
            if self.mode == "crash" or (self.mode == "crash_first" and line_number == 1):
                self.returncode = 1
                self.responses.put("")
                return
            if self.mode == "no_response":
                continue
            if self.response_delay:
                time.sleep(self.response_delay)
            if self.mode == "malformed":
                self.responses.put("not-json\n")
                continue
            self.responses.put(json.dumps(make_response(self.received[-1])) + "\n")

    def poll(self):
        return self.returncode

    def terminate(self):
        if self.returncode is None:
            self.returncode = -15
        self.requests.put(None)
        self.responses.put("")

    def kill(self):
        if self.returncode is None:
            self.returncode = -9
        self.requests.put(None)
        self.responses.put("")

    def wait(self, timeout=None):
        self._thread.join(timeout=timeout)
        if self.returncode is None:
            raise TimeoutError("fake process did not stop")
        return self.returncode


def make_response(payload):
    transaction_id = payload.get("request_id", 50)
    return {
        "algorithm": "priority_free_variable_dqn",
        "action_trace": [transaction_id],
        "steps": 1,
        "completed_transaction_ids": [transaction_id],
        "cpp_trace_records": [f"BURST,{transaction_id},1", f"ACTION,{transaction_id}"],
        "request_id": payload.get("request_id"),
    }


def make_settings(tmp_path, *, timeout=0.5):
    python = tmp_path / "python.exe"
    checkpoint = tmp_path / "priority-free.pt"
    cpp = tmp_path / "replay.exe"
    adapter = tmp_path / "priority_free_inference_adapter.py"
    for path in (python, checkpoint, cpp, adapter):
        path.write_text("present", encoding="utf-8")
    return SimpleNamespace(
        drl_python_path=python,
        priority_free_dqn_checkpoint_path=checkpoint,
        priority_free_dqn_cpp_engine_path=cpp,
        priority_free_dqn_adapter_path=adapter,
        simulation_timeout_seconds=timeout,
    )


class FakePopenFactory:
    def __init__(self, modes=(), *, response_delay=0):
        self.modes = list(modes)
        self.response_delay = response_delay
        self.processes = []
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        mode = self.modes.pop(0) if self.modes else "normal"
        process = FakeProcess(mode, response_delay=self.response_delay)
        self.processes.append(process)
        return process


def create_manager(tmp_path, factory, *, timeout=0.5):
    settings = make_settings(tmp_path, timeout=timeout)
    manager = PriorityFreeDqnWorkerManager(
        settings_provider=lambda: settings,
        popen_factory=factory,
    )
    return manager, settings


def test_worker_starts_lazily_and_reuses_one_process(tmp_path):
    factory = FakePopenFactory()
    manager, settings = create_manager(tmp_path, factory)
    assert factory.calls == []

    first = manager.request({"request_id": 1}, settings=settings)
    second = manager.request({"request_id": 2}, settings=settings)

    assert len(factory.calls) == 1
    command, kwargs = factory.calls[0]
    assert command == [
        str(settings.drl_python_path), "-m", "drl_scheduler.priority_free_inference_adapter",
        "--worker", "--checkpoint", str(settings.priority_free_dqn_checkpoint_path),
    ]
    assert kwargs["stderr"] is None
    assert first["request_id"] == 1
    assert second["request_id"] == 2
    assert factory.processes[0].received == [{"request_id": 1}, {"request_id": 2}]
    manager.shutdown()


def test_concurrent_requests_are_serialized_without_lost_or_mixed_responses(tmp_path):
    factory = FakePopenFactory(response_delay=0.02)
    manager, settings = create_manager(tmp_path, factory)
    results = {}
    failures = []

    def send(request_id):
        try:
            results[request_id] = manager.request({"request_id": request_id}, settings=settings)
        except Exception as error:  # surfaced after all client threads join
            failures.append(error)

    threads = [threading.Thread(target=send, args=(number,)) for number in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert not failures
    assert all(not thread.is_alive() for thread in threads)
    assert len(factory.processes) == 1
    assert len(factory.processes[0].received) == 10
    assert {request_id: response["request_id"] for request_id, response in results.items()} == {
        number: number for number in range(10)
    }
    manager.shutdown()


def test_worker_crash_is_reported_and_next_request_restarts_it(tmp_path):
    factory = FakePopenFactory(["crash", "normal"])
    manager, settings = create_manager(tmp_path, factory)

    with pytest.raises(WorkerUnavailableError):
        manager.request({"request_id": 1}, settings=settings)
    recovered = manager.request({"request_id": 2}, settings=settings)

    assert recovered["request_id"] == 2
    assert len(factory.processes) == 2
    manager.shutdown()


def test_broken_worker_pipe_is_reported_and_next_request_restarts_it(tmp_path):
    factory = FakePopenFactory(["broken_pipe", "normal"])
    manager, settings = create_manager(tmp_path, factory)

    with pytest.raises(WorkerUnavailableError, match="input pipe"):
        manager.request({"request_id": 1}, settings=settings)
    recovered = manager.request({"request_id": 2}, settings=settings)

    assert recovered["request_id"] == 2
    assert len(factory.processes) == 2
    manager.shutdown()


def test_worker_timeout_stops_old_process_and_next_request_restarts(tmp_path):
    factory = FakePopenFactory(["no_response", "normal"])
    manager, settings = create_manager(tmp_path, factory, timeout=0.02)

    with pytest.raises(WorkerTimeoutError):
        manager.request({"request_id": 1}, settings=settings)
    recovered = manager.request({"request_id": 2}, settings=settings)

    assert factory.processes[0].returncode is not None
    assert recovered["request_id"] == 2
    assert len(factory.processes) == 2
    manager.shutdown()


def test_malformed_worker_response_discards_process_and_restarts(tmp_path):
    factory = FakePopenFactory(["malformed", "normal"])
    manager, settings = create_manager(tmp_path, factory)

    with pytest.raises(WorkerProtocolError, match="invalid JSON"):
        manager.request({"request_id": 1}, settings=settings)
    response = manager.request({"request_id": 2}, settings=settings)

    assert response["request_id"] == 2
    assert len(factory.processes) == 2
    manager.shutdown()


def test_shutdown_terminates_worker_and_rejects_later_requests(tmp_path):
    factory = FakePopenFactory()
    manager, settings = create_manager(tmp_path, factory)
    manager.request({"request_id": 1}, settings=settings)
    process = factory.processes[0]

    manager.shutdown()

    assert process.returncode is not None
    with pytest.raises(WorkerUnavailableError, match="shut down"):
        manager.request({"request_id": 2}, settings=settings)
    manager.resume()
    assert manager.request({"request_id": 3}, settings=settings)["request_id"] == 3
    assert len(factory.processes) == 2
    manager.shutdown()


def test_fastapi_lifespan_shuts_down_the_worker_manager(monkeypatch):
    calls = []
    monkeypatch.setattr(main_module.priority_free_dqn_worker, "resume", lambda: calls.append("resume"))
    monkeypatch.setattr(main_module.priority_free_dqn_worker, "shutdown", lambda: calls.append("shutdown"))

    with TestClient(app):
        pass

    assert calls == ["resume", "shutdown"]
