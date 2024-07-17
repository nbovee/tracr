import tracr
import tracr.app_api.utils as utils
from importlib.util import find_spec


def test_import():
    try:
        find_spec("tracr")

        print("Import successful")
    except ImportError as e:
        print(f"Import failed: {e}")


def test_version():
    try:
        version = tracr.__version__
        print(f"tracr version: {version}")
        assert version == "0.1"
    except AttributeError as e:
        print(f"Version attribute missing: {e}")


def test_functions():
    try:
        ip = utils.get_local_ip()
        print(f"Local IP: {ip}")

        repo_root = utils.get_repo_root()
        print(f"Repo root: {repo_root}")

        registery_up = utils.registry_server_is_up()
        print(f"Register server is up: {registery_up}")
    except AttributeError as e:
        print(f"Function missing: {e}")


if __name__ == "__main__":
    test_import()
    test_version()
    test_functions()
