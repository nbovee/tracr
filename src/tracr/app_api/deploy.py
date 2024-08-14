import logging
import os
import rpyc
import sys
import shutil
import tempfile
import plumbum
from rpyc.utils.zerodeploy import DeployedServer, TimeoutExpired
from rpyc.core.stream import SocketStream
from plumbum.machines.session import ShellSessionError
from plumbum.machines.local import LocalMachine
from plumbum.machines.remote import RemoteCommand
from plumbum import SshMachine, local, CommandNotFound
from plumbum.path import copy
from plumbum.commands.base import BoundCommand

from . import device_mgmt as dm
from . import utils

logger = logging.getLogger("tracr_logger")


class ZeroDeployedServer(DeployedServer):
    """
    A class for deploying and managing a remote RPyC server with zero-configuration.
    This class extends the DeployedServer class from RPyC.
    """

    def __init__(
        self,
        device: dm.Device,
        node_name: str,
        model: tuple[str, str],
        participant_service: tuple[str, str],
        server_class="rpyc.utils.server.ThreadedServer",
        python_executable=None,
        timeout_s: int = 600,
    ):
        """
        Initialize the ZeroDeployedServer.

        This method sets up the remote environment, copies necessary files,
        and starts the RPyC server on the remote machine.

        Args:
            device (dm.Device): The device to deploy the server on.
            node_name (str): The name of the node.
            model (tuple): The model module and class names.
            participant_service (tuple): The participant service module and class names.
            server_class (str): The RPyC server class to use.
            python_executable (str): The Python executable to use on the remote machine.
            timeout_s (int): The maximum uptime for the server in seconds.
        """
        logger.debug(f"Constructing ZeroDeployedServer for {node_name}.")
        assert device.working_cparams is not None
        self.name = device._name
        self.proc = None
        self.remote_machine = device.as_pb_sshmachine()
        self._tmpdir_ctx = None

        # Handle local and remote machines differently
        if isinstance(self.remote_machine, LocalMachine):
            tmp = local.path(local.env.get('TEMP', '/tmp')) / f"tracr_tmp_{node_name}"
            tmp.mkdir(exist_ok=True)
        else:
            self._tmpdir_ctx = self.remote_machine.tempdir()
            tmp = self._tmpdir_ctx.__enter__()

        # Create necessary directories
        if isinstance(self.remote_machine, LocalMachine):
            os.makedirs(os.path.join(tmp, "src", "tracr"), exist_ok=True)
        else:
            self.remote_machine["mkdir"]["-p", f"{tmp}/src/tracr"] & plumbum.FG

        # Copy over the rpyc and experiment_design packages
        rpyc_root = local.path(rpyc.__file__).up()
        logger.debug(f"Copying rpyc from {rpyc_root} to {tmp / 'rpyc'}")
        self.safe_copy(rpyc_root, tmp / "rpyc")

        src_root = local.path(utils.get_repo_root() / "src" / "tracr")
        logger.debug(f"Copying src from {src_root} to {tmp / 'src' / 'tracr'}")
        self.safe_copy(src_root, tmp / "src" / "tracr")

        # Substitute placeholders in the remote script and send it over
        script = tmp / "deployed-rpyc.py"
        modname, clsname = server_class.rsplit(".", 1)
        m_module, m_class = model
        logger.debug(f"Model module: {m_module}")
        logger.debug(f"Model class: {m_class}")
        ps_module, ps_class = participant_service
        observer_ip = utils.get_local_ip()
        participant_host = device.working_cparams.host

        # Load the server script template
        with open(utils.get_repo_root() / "src" / "tracr" / "app_api" / "server_script.py", "r") as f:
            SERVER_SCRIPT = f.read()

        script_content = (
            SERVER_SCRIPT.replace("$SVR-MODULE$", modname)
            .replace("$SVR-CLASS$", clsname)
            .replace("$MOD-MODULE$", m_module)
            .replace("$MOD-CLASS$", m_class)
            .replace("$PS-MODULE$", ps_module)
            .replace("$PS-CLASS$", ps_class)
            .replace("$NODE-NAME$", node_name)
            .replace("$OBS-IP$", observer_ip)
            .replace("$PRT-HOST$", participant_host)
            .replace("$MAX-UPTIME$", f"{timeout_s}")
        )
        
        self.write_script(script, script_content)
        cmd = self._determine_python_executable(python_executable)

        if isinstance(self.remote_machine, LocalMachine):
            self.proc = cmd.popen(str(script), new_session=True)
        else:
            assert isinstance(cmd, RemoteCommand)
            logger.debug(f"Starting remote process for {node_name}")
            self.proc = cmd.popen(script, new_session=True)
            logger.debug(f"Remote process started for {node_name}")

        logger.debug(f"ZeroDeployedServer initialization completed for {node_name}")

    def write_script(self, script_path, content):
        if isinstance(self.remote_machine, LocalMachine):
            with open(script_path, 'w') as f:
                f.write(content)
        else:
            # For remote machines, use a temporary local file
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                temp_file.write(content)
                temp_file.flush()
                temp_file_path = temp_file.name

            try:
                self.remote_machine.upload(temp_file_path, script_path)
            finally:
                os.unlink(temp_file_path)

    def safe_copy(self, src, dst):
        try:
            if isinstance(self.remote_machine, LocalMachine):
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            else:
                # For remote machines, use SCP
                self.remote_machine.upload(src, dst)
        except Exception as e:
            logger.error(f"Error copying {src} to {dst}: {str(e)}")
            raise

    def _determine_python_executable(self, python_executable):
        """
        Determines the appropriate Python executable to use on the remote machine.

        Args:
            python_executable (str or BoundCommand): The specified Python executable.

        Returns:
            RemoteCommand: The determined Python executable command.
        """
        if isinstance(python_executable, BoundCommand):
            return python_executable
        if python_executable:
            return self.remote_machine[python_executable]

        major, minor = sys.version_info[:2]
        logger.info(
            f"Observer uses Python {major}.{minor}. Looking for equivalent Python executable on {self.name}"
        )

        for opt in [f"python{major}.{minor}", f"python{major}"]:
            try:
                logger.info(f"Checking {opt}")
                cmd = self.remote_machine[opt]
                logger.info(f"{opt} is available.")
                return cmd
            except CommandNotFound:
                logger.info(f"{opt} is not available.")

        logger.warning(
            "Using the default python interpreter, which could cause problems."
        )
        return self.remote_machine["python"]

    def _connect_sock(self, port=18861):
        """
        Connects to the remote socket stream.

        Returns:
            SocketStream: The connected socket stream.
        """
        assert isinstance(self.remote_machine, SshMachine)
        return SocketStream._connect(self.remote_machine.host, port)

    def __del__(self):
        """
        Destructor to ensure proper cleanup of resources.
        """
        try:
            super().__del__()
        except (AttributeError, ShellSessionError):
            pass

    def close(self, timeout=5):
        if hasattr(self.remote_machine, '_session') and self.remote_machine._session:
            self._terminate_process(self.remote_machine._session.proc, timeout)
        else:
            # Handle the case where _session doesn't exist
            logger.warning(f"No active session found for {self.node_name}. Skipping process termination.")
        
        if self.remote_machine:
            self.remote_machine.close()
        self.remote_machine = None

    def close(self, timeout=5):
        """
        Closes the deployed server and cleans up resources.

        Args:
            timeout (int, optional): The timeout for closing operations. Defaults to 5.
        """
        if hasattr(self, 'proc'):
            self._terminate_process(self.proc, timeout)
            self.proc = None

        if hasattr(self, 'remote_machine') and self.remote_machine:
            if hasattr(self.remote_machine, '_session'):
                self._terminate_process(self.remote_machine._session.proc, timeout)
            if hasattr(self.remote_machine, 'close'):
                try:
                    self.remote_machine.close()
                except Exception as e:
                    logger.error(f"Error closing remote machine: {str(e)}")
            self.remote_machine = None

        if hasattr(self, '_tmpdir_ctx') and self._tmpdir_ctx:
            try:
                self._tmpdir_ctx.__exit__(None, None, None)
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {str(e)}")
            self._tmpdir_ctx = None

        logger.info(f"Closed ZeroDeployedServer for {getattr(self, 'node_name', 'unknown node')}")

    def _terminate_process(self, proc, timeout):
        """
        Terminates the given process with a timeout.

        Args:
            proc (subprocess.Popen): The process to terminate.
            timeout (int): The timeout for termination.
        """
        if proc:
            try:
                proc.terminate()
                proc.communicate(timeout=timeout)
            except TimeoutExpired:
                proc.kill()
                raise
            except Exception:
                pass
