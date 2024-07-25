import rpyc
import logging
import sys
from rpyc.utils.zerodeploy import DeployedServer, TimeoutExpired
from rpyc.core.stream import SocketStream
from plumbum.machines.session import ShellSessionError
from plumbum.machines.remote import RemoteCommand
from plumbum import SshMachine, local, CommandNotFound
from plumbum.path import copy
from plumbum.commands.base import BoundCommand

from src.tracr.app_api import device_mgmt as dm
from src.tracr.app_api import utils
from src.tracr.app_api.server_script import SERVER_SCRIPT

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

        # Create a temp dir on the remote machine where we make the environment
        self._tmpdir_ctx = self.remote_machine.tempdir()
        tmp = self._tmpdir_ctx.__enter__()

        # Copy over the rpyc and experiment_design packages
        rpyc_root = local.path(rpyc.__file__).up()
        copy(rpyc_root, tmp / "rpyc")

        src_root = local.path(utils.get_repo_root() / "src" / "tracr")
        copy(src_root, tmp / "src" / "tracr")

        # Substitute placeholders in the remote script and send it over
        script = tmp / "deployed-rpyc.py"
        modname, clsname = server_class.rsplit(".", 1)
        m_module, m_class = model
        ps_module, ps_class = participant_service
        observer_ip = utils.get_local_ip()
        participant_host = device.working_cparams.host
        script.write(
            SERVER_SCRIPT.replace("$SVR-MODULE$", modname)
            .replace("$SVR-CLASS$", clsname)
            .replace("$MOD-MODULE$", m_module)
            .replace("$MOD-CLASS$", m_class)
            .replace("$PS-MODULE$", ps_module)
            .replace("$PS-CLASS$", ps_class)
            .replace("$NODE-NAME$", node_name)
            .replace("$OBS-IP$", observer_ip)
            .replace("$PRT-HOST$", participant_host)
            .replace("$MAX-UPTIME$", str(timeout_s))
        )

        cmd = self._determine_python_executable(python_executable)
        assert isinstance(cmd, RemoteCommand)
        self.proc = cmd.popen(script, new_session=True)

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
        return self.remote_machine.python

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
        """
        Closes the deployed server and cleans up resources.

        Args:
            timeout (int, optional): The timeout for closing operations. Defaults to 5.
        """
        self._terminate_process(self.proc, timeout)
        self.proc = None

        if self.remote_machine:
            self._terminate_process(self.remote_machine._session.proc, timeout)
            self.remote_machine.close()
            self.remote_machine = None

        if self._tmpdir_ctx:
            try:
                self._tmpdir_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._tmpdir_ctx = None

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
