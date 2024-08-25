import logging
import os
import rpyc
import sys
import shutil
import tempfile
import plumbum
import subprocess
import threading
from rpyc.utils.zerodeploy import DeployedServer, TimeoutExpired
from rpyc.core.stream import SocketStream
from plumbum.machines.local import LocalMachine
from plumbum.machines.remote import RemoteCommand
from plumbum import SshMachine, local, CommandNotFound
from plumbum.commands.base import BoundCommand
from typing import Optional, Union, Tuple

from . import device_mgmt as dm
from . import utils

logger = logging.getLogger("tracr_logger")


class ZeroDeployedServer(DeployedServer):
    """A class for deploying and managing a remote RPyC server with zero-configuration."""

    def __init__(
        self,
        device: dm.Device,
        node_name: str,
        model: Tuple[str, str],
        participant_service: Tuple[str, str],
        server_class: str = "rpyc.utils.server.ThreadedServer",
        python_executable: Optional[Union[str, BoundCommand]] = None,
        timeout_s: int = 600,
    ):
        """Initialize the ZeroDeployedServer."""
        logger.debug(f"Constructing ZeroDeployedServer for {node_name}.")
        if device.working_cparams is None:
            raise ValueError(
                f"Device {device._name} has no working connection parameters")

        self.name = device._name
        self.node_name = node_name
        self.proc: Optional[subprocess.Popen] = None
        self.remote_machine: Union[LocalMachine,
                                   SshMachine] = device.as_pb_sshmachine()
        self._tmpdir_ctx: Optional[plumbum.path.local.LocalPath] = None

        try:
            tmp = self._setup_temporary_directory()
            self._copy_necessary_files(tmp)
            script = self._prepare_server_script(
                tmp, server_class, model, participant_service, timeout_s)
            cmd = self._determine_python_executable(python_executable)
            self._start_server_process(cmd, script)
        except Exception as e:
            logger.error(
                f"Error initializing ZeroDeployedServer for {node_name}: {str(e)}")
            self.close()
            raise

        logger.debug(
            f"ZeroDeployedServer initialization completed for {node_name}")

    def _setup_temporary_directory(self) -> plumbum.path.base.Path:
        """Set up the temporary directory for the server."""
        if isinstance(self.remote_machine, LocalMachine):
            tmp = local.path(local.env.get("TEMP", "/tmp")) / \
                f"tracr_tmp_{self.node_name}"
            tmp.mkdir(exist_ok=True)
        else:
            self._tmpdir_ctx = self.remote_machine.tempdir()
            tmp = self._tmpdir_ctx.__enter__()

        if isinstance(self.remote_machine, LocalMachine):
            os.makedirs(os.path.join(tmp, "src", "tracr"), exist_ok=True)
        else:
            self.remote_machine["mkdir"]["-p", f"{tmp}/src/tracr"] & plumbum.FG

        return tmp

    def _copy_necessary_files(self, tmp: plumbum.path.base.Path) -> None:
        """Copy necessary files to the temporary directory."""
        rpyc_root = local.path(rpyc.__file__).up()
        logger.debug(f"Copying rpyc from {rpyc_root} to {tmp / 'rpyc'}")
        self.safe_copy(rpyc_root, tmp / "rpyc")

        src_root = local.path(utils.get_repo_root() / "src" / "tracr")
        target_dir = tmp / "src" / "tracr"
        logger.debug(
            f"Copying tracr from {src_root} to {target_dir}")

        # Instead of copying the entire tracr folder into tracr again, copy its contents
        if isinstance(self.remote_machine, LocalMachine):
            shutil.copytree(src_root, target_dir, dirs_exist_ok=True)
        else:
            # Ensure target directory exists before copying
            self.remote_machine["mkdir"]["-p", target_dir] & plumbum.FG
            for item in src_root:
                self.safe_copy(item, target_dir / item.name)

    def _prepare_server_script(
        self,
        tmp: plumbum.path.base.Path,
        server_class: str,
        model: Tuple[str, str],
        participant_service: Tuple[str, str],
        timeout_s: int
    ) -> plumbum.path.base.Path:
        """Prepare the server script with the necessary configurations."""
        script = tmp / "deployed-rpyc.py"
        modname, clsname = server_class.rsplit(".", 1)
        m_module, m_class = model
        ps_module, ps_class = participant_service
        observer_ip = utils.get_local_ip()

        # Handle local and remote machines differently
        if isinstance(self.remote_machine, LocalMachine):
            participant_host = "localhost"
        else:
            participant_host = self.remote_machine.host

        with open(utils.get_repo_root() / "src" / "tracr" / "app_api" / "server_script.py", "r") as f:
            SERVER_SCRIPT = f.read()

        script_content = (
            SERVER_SCRIPT.replace("$SVR-MODULE$", modname)
            .replace("$SVR-CLASS$", clsname)
            .replace("$MOD-MODULE$", m_module)
            .replace("$MOD-CLASS$", m_class)
            .replace("$PS-MODULE$", ps_module)
            .replace("$PS-CLASS$", ps_class)
            .replace("$NODE-NAME$", self.node_name)
            .replace("$OBS-IP$", observer_ip)
            .replace("$PRT-HOST$", participant_host)
            .replace("$MAX-UPTIME$", f"{timeout_s}")
        )

        self.write_script(script, script_content)
        return script

    def _start_server_process(self, cmd: BoundCommand, script: plumbum.path.base.Path) -> None:
        """Start the server process on the remote machine."""
        if isinstance(self.remote_machine, LocalMachine):
            self.proc = cmd.popen(
                str(script), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            assert isinstance(cmd, RemoteCommand)
            logger.debug(f"Starting remote process for {self.node_name}")
            self.proc = cmd.popen(
                script, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.debug(f"Remote process started for {self.node_name}")

        # Start threads to continuously read and log the output
        def log_output(stream, log_func):
            for line in iter(stream.readline, b''):
                log_func(f"{self.node_name} output: {line.decode().strip()}")

        threading.Thread(target=log_output, args=(
            self.proc.stdout, logger.info), daemon=True).start()
        threading.Thread(target=log_output, args=(
            self.proc.stderr, logger.error), daemon=True).start()

    def write_script(self, script_path: plumbum.path.base.Path, content: str) -> None:
        """Write the server script to the remote machine."""
        if isinstance(self.remote_machine, LocalMachine):
            with open(script_path, "w") as f:
                f.write(content)
        else:
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                temp_file.write(content)
                temp_file.flush()
                temp_file_path = temp_file.name

            try:
                self.remote_machine.upload(temp_file_path, script_path)
            finally:
                os.unlink(temp_file_path)

    def safe_copy(self, src: Union[str, plumbum.path.base.Path], dst: plumbum.path.base.Path) -> None:
        """Safely copy files or directories to the remote machine."""
        try:
            if isinstance(self.remote_machine, LocalMachine):
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            else:
                self.remote_machine.upload(src, dst)
        except Exception as e:
            logger.error(f"Error copying {src} to {dst}: {str(e)}")
            raise

    def _determine_python_executable(self, python_executable: Optional[Union[str, BoundCommand]]) -> BoundCommand:
        """Determine the appropriate Python executable to use on the remote machine."""
        if isinstance(python_executable, BoundCommand):
            return python_executable
        if python_executable:
            return self.remote_machine[python_executable]

        major, minor = sys.version_info[:2]
        logger.info(
            f"Observer uses Python {major}.{minor}. Looking for equivalent Python executable on {self.name}")

        for opt in [f"python{major}.{minor}", f"python{major}", "python3", "python"]:
            try:
                logger.info(f"Checking {opt}")
                cmd = self.remote_machine[opt]
                logger.info(f"{opt} is available.")
                return cmd
            except CommandNotFound:
                logger.info(f"{opt} is not available.")

        raise RuntimeError(
            f"No suitable Python interpreter found on {self.name}")

    def _connect_sock(self, port: int = 18861) -> SocketStream:
        """Connect to the remote socket stream."""
        if not isinstance(self.remote_machine, SshMachine):
            raise TypeError(
                "Remote machine must be an SshMachine for socket connection")
        return SocketStream._connect(self.remote_machine.host, port)

    def close(self, timeout: int = 5) -> None:
        """Close the deployed server and clean up resources."""
        if hasattr(self, "proc") and self.proc:
            self._terminate_process(self.proc, timeout)
            self.proc = None

        if hasattr(self, "remote_machine") and self.remote_machine:
            if hasattr(self.remote_machine, "_session"):
                self._terminate_process(
                    self.remote_machine._session.proc, timeout)
            if hasattr(self.remote_machine, "close"):
                try:
                    self.remote_machine.close()
                except Exception as e:
                    logger.error(f"Error closing remote machine: {str(e)}")
            self.remote_machine = None

        if self._tmpdir_ctx:
            try:
                self._tmpdir_ctx.__exit__(None, None, None)
            except Exception as e:
                logger.error(
                    f"Error cleaning up temporary directory: {str(e)}")
            self._tmpdir_ctx = None

        logger.info(f"Closed ZeroDeployedServer for {self.node_name}")

    def _terminate_process(self, proc: subprocess.Popen, timeout: int) -> None:
        """Terminate the given process with a timeout."""
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=timeout)
            except TimeoutExpired:
                logger.warning(
                    f"Process termination timed out for {self.node_name}, forcing kill")
                proc.kill()
            except Exception as e:
                logger.error(
                    f"Error terminating process for {self.node_name}: {str(e)}")

    def __del__(self) -> None:
        """Ensure proper cleanup of resources on object deletion."""
        try:
            self.close()
        except Exception as e:
            logger.error(f"Error in __del__ for {self.node_name}: {str(e)}")
