import importlib

from lionz import system


def test_models_directory_can_be_configured_from_the_environment(monkeypatch):
    configured_directory = "/usr/local/models/nnunet_trained_models"

    with monkeypatch.context() as environment:
        environment.setenv("LIONZ_MODELS_DIRECTORY", configured_directory)
        reloaded_system = importlib.reload(system)

        assert reloaded_system.MODELS_DIRECTORY_PATH == configured_directory

    importlib.reload(system)
