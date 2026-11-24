from pathlib import Path

from dashboard.backend.app.config import (
    BACKEND_DIR,
    DEFAULT_DQN_BRIDGE_PATH,
    DEFAULT_DQN_CHECKPOINT_PATH,
    DEFAULT_DQN_CPP_ENGINE_PATH,
    DEFAULT_DRL_PYTHON_PATH,
    DEFAULT_PRIORITY_FREE_DQN_CHECKPOINT_PATH,
    DEFAULT_PRIORITY_FREE_DQN_CPP_ENGINE_PATH,
    DEFAULT_RESOURCE_DEMO_CPP_ENGINE_PATH,
    REPOSITORY_ROOT,
    get_settings,
)


def test_cpp_executable_defaults_to_cli_build(monkeypatch):
    monkeypatch.delenv("CPP_ENGINE_PATH", raising=False)

    settings = get_settings()

    expected = (
        REPOSITORY_ROOT
        / "core_engine"
        / "build"
        / "cli-build"
        / "IntelligentResourceManager.exe"
    ).resolve()
    assert settings.cpp_engine_path == expected


def test_cpp_engine_path_environment_variable_overrides_default(monkeypatch, tmp_path):
    override = tmp_path / "custom-engine.exe"
    monkeypatch.setenv("CPP_ENGINE_PATH", str(override))

    settings = get_settings()

    assert settings.cpp_engine_path == override.resolve()


def test_relative_cpp_engine_path_is_resolved_from_backend_directory(monkeypatch):
    relative_override = Path("custom") / "Scheduler.exe"
    monkeypatch.setenv("CPP_ENGINE_PATH", str(relative_override))

    settings = get_settings()

    assert settings.cpp_engine_path == (BACKEND_DIR / relative_override).resolve()


def test_dqn_runtime_paths_have_project_relative_defaults(monkeypatch):
    for name in ("DRL_PYTHON_PATH", "DQN_CPP_ENGINE_PATH", "DQN_CHECKPOINT_PATH"):
        monkeypatch.delenv(name, raising=False)

    settings = get_settings()

    assert settings.drl_python_path == DEFAULT_DRL_PYTHON_PATH.resolve()
    assert settings.dqn_cpp_engine_path == DEFAULT_DQN_CPP_ENGINE_PATH.resolve()
    assert settings.dqn_bridge_path == DEFAULT_DQN_BRIDGE_PATH.resolve()
    assert settings.dqn_checkpoint_path == DEFAULT_DQN_CHECKPOINT_PATH.resolve()


def test_dqn_runtime_paths_allow_environment_overrides(monkeypatch, tmp_path):
    python = tmp_path / "python.exe"
    engine = tmp_path / "replay.exe"
    checkpoint = tmp_path / "checkpoint.pt"
    monkeypatch.setenv("DRL_PYTHON_PATH", str(python))
    monkeypatch.setenv("DQN_CPP_ENGINE_PATH", str(engine))
    monkeypatch.setenv("DQN_CHECKPOINT_PATH", str(checkpoint))

    settings = get_settings()

    assert settings.drl_python_path == python.resolve()
    assert settings.dqn_cpp_engine_path == engine.resolve()
    assert settings.dqn_checkpoint_path == checkpoint.resolve()


def test_priority_free_dqn_runtime_paths_have_separate_defaults(monkeypatch):
    monkeypatch.delenv("PRIORITY_FREE_DQN_CPP_ENGINE_PATH", raising=False)
    monkeypatch.delenv("PRIORITY_FREE_DQN_CHECKPOINT_PATH", raising=False)

    settings = get_settings()

    assert settings.priority_free_dqn_cpp_engine_path == DEFAULT_PRIORITY_FREE_DQN_CPP_ENGINE_PATH.resolve()
    assert settings.priority_free_dqn_checkpoint_path == DEFAULT_PRIORITY_FREE_DQN_CHECKPOINT_PATH.resolve()
    assert settings.priority_free_dqn_checkpoint_path != settings.dqn_checkpoint_path


def test_priority_free_dqn_runtime_paths_allow_independent_overrides(monkeypatch, tmp_path):
    executable = tmp_path / "priority-free-replay.exe"
    checkpoint = tmp_path / "priority-free-checkpoint.pt"
    monkeypatch.setenv("PRIORITY_FREE_DQN_CPP_ENGINE_PATH", str(executable))
    monkeypatch.setenv("PRIORITY_FREE_DQN_CHECKPOINT_PATH", str(checkpoint))

    settings = get_settings()

    assert settings.priority_free_dqn_cpp_engine_path == executable.resolve()
    assert settings.priority_free_dqn_checkpoint_path == checkpoint.resolve()


def test_resource_demo_cli_has_its_own_default_and_environment_override(monkeypatch, tmp_path):
    monkeypatch.delenv("RESOURCE_DEMO_CPP_ENGINE_PATH", raising=False)
    settings = get_settings()
    assert settings.resource_demo_cpp_engine_path == DEFAULT_RESOURCE_DEMO_CPP_ENGINE_PATH.resolve()

    override = tmp_path / "resource-demo.exe"
    monkeypatch.setenv("RESOURCE_DEMO_CPP_ENGINE_PATH", str(override))
    assert get_settings().resource_demo_cpp_engine_path == override.resolve()
