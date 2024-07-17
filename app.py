#!/usr/bin/env python3

"""
This script runs the "tracr" CLI, which is powered by the API that lives in
the "api" folder on this repo.

For a cleaner experience, add this directory to your PATH, which will allow
you to run the CLI from anywhere, and without preceding the command with
the word "python".
"""

import logging


import argparse
from rich.console import Console
from rich.table import Table
from time import sleep

from src.tracr.app_api import log_handling, utils
from src.tracr.app_api.device_mgmt import DeviceMgr
from src.tracr.app_api.experiment_mgmt import Experiment, ExperimentManifest

logger = logging.getLogger("tracr_logger")

PROJECT_ROOT = utils.get_repo_root()
CURRENT_VERSION = "0.3.0"


logger = log_handling.setup_logging()


# CLI is split up into submodules responsible for different operations, so
# there are lots of arguments that can be used, each with their own option
# flags and arguments. This is all organized using argparse, which reads
# the arguments and options and passes them to the appropriate function.


##############################################################################
######################### SETUP SUBMODULE FUNCTIONS ##########################
##############################################################################


def setup_controller(args):
    pass


def setup_device(args):
    # TODO: define setup device function
    pass


def setup_experiment(args):
    # TODO: define setup experiment function
    pass


##############################################################################
##################### DEVICE SUBMODULE FUNCTIONS #############################
##############################################################################


def device_ls(args):
    """
    Shows the user a list of devices.
    Called by the "tracr device ls [-options]" command.

    Parameters:
    -----------
    args: argparse.Namespace
        The arguments and options passed to the CLI.
    """
    device_mgr = DeviceMgr()
    devices = device_mgr.get_devices()
    console = Console()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name")
    table.add_column("Reachable")
    table.add_column("Ready")
    table.add_column("Host")
    table.add_column("User")

    for d in devices:
        name = d._name
        can_be_reached = d.is_reachable()
        reachable = (
            "[bold green]Yes[/bold green]"
            if can_be_reached
            else "[bold red]No[/bold red]"
        )
        ready = (
            "[bold green]Yes[/bold green]"
            if can_be_reached  # False
            else "[bold red]No[/bold red]"
        )
        host = d.get_current("host")
        user = d.get_current("user")

        table.add_row(name, reachable, ready, host, user)

    console.print(table)


def device_add(args):
    if args.wizard:
        pass
    if args.host:
        pass
    if args.user:
        pass
    if args.pw:
        pass
    if args.keys:
        pass
    if args.nickname:
        pass
    if args.description:
        pass


##############################################################################
#################### EXPERIMENT SUBMODULE FUNCTIONS ##########################
##############################################################################


def experiment_add(args):
    """
    Adds a new experiment to the controller's local system by setting up
    a 'blank slate' directory inside of TestCases.
    """
    pass


def experiment_ls(args):
    """
    Displays the list of experiments that are currently available.
    """
    pass


def experiment_run(args):
    """
    Runs an experiment.
    """
    exp_name = args.name
    logger.info(f"Attempting to set up experiment {exp_name}.")
    testcase_dir = PROJECT_ROOT / "UserData" / "TestCases"
    manifest_yaml_fp = next(testcase_dir.glob(f"**/*{exp_name}.yaml"))
    logger.debug(f"Found manifest at {str(manifest_yaml_fp)}.")
    rlog_server = log_handling.get_server_running_in_thread()
    manifest = ExperimentManifest(manifest_yaml_fp)

    logger.debug("Initializing DeviceMgr object.")
    device_manager = DeviceMgr()
    available_devices = device_manager.get_devices(available_only=True)

    logger.debug("Initializing Experiment object.")
    experiment = Experiment(manifest, available_devices)
    logger.debug(f"Running experiment {exp_name}.")
    experiment.run()

    log_handling.shutdown_gracefully(rlog_server)
    sleep(2)  # give the remaining remote logs a second to be displayed
    logger.info("Congratulations! The experiment has concluded successfully.")


def network(args):
    if args.d:
        print(f"Status for host: {args.d}")
    elif args.e:
        print(f"Status for name: {args.e}")
    else:
        print("Running network")


def run(args):
    if args.e:
        print(f"Running with name: {args.e}")
    else:
        print("Running")


def setup(args):
    if args.d:
        print(f"Setup for host: {args.d}")
    elif args.e:
        print(f"Setup for name: {args.e}")
    else:
        print("Running setup")


def main():
    parser = argparse.ArgumentParser(
        description=r""" | |
 | |_ _ __ __ _  ___ _ __
 | __| '__/ _` |/ __| '__|
 | |_| | | (_| | (__| |
  \__|_|  \__,_|\___|_|

A CLI for conducting collaborative AI experiments.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=CURRENT_VERSION,
    )

    subparsers = parser.add_subparsers(title="SUBMODULES")

    # parser for "device"
    parser_device = subparsers.add_parser(
        "device", help="Local device setup and configuration"
    )
    # add sub-sub parsers for the device module
    device_subparsers = parser_device.add_subparsers(title="DEVICE MODULE COMMANDS")
    parser_device_ls = device_subparsers.add_parser("ls", help="List devices")
    parser_device_ls.set_defaults(func=device_ls)

    parser_device_add = device_subparsers.add_parser(
        "add", help="add new devices and configure them for experiments"
    )
    parser_device_add.add_argument(
        "-w",
        "--wizard",
        action="store_true",
        help="use a wizard to help with adding the new device",
        dest="wizard",
    )
    parser_device_add.add_argument(
        "-a",
        "--host",
        help="specify the hostname or IP address of the device to add",
        nargs=1,
        dest="host",
    )
    parser_device_add.add_argument(
        "-u",
        "--user",
        help="specify the username to connect with via SSH",
        nargs=1,
        dest="user",
    )
    parser_device_add.add_argument(
        "-k",
        "--keys",
        help="specify the public and private keys, separated by a space",
        nargs=2,
        dest="keys",
    )
    parser_device_add.add_argument(
        "-n",
        "--nickname",
        help="assign a nickname to the device",
        nargs=1,
        dest="nickname",
    )
    parser_device_add.add_argument(
        "-d",
        "--description",
        help="give a description to the device",
        nargs=1,
        dest="description",
    )
    parser_device_add.set_defaults(func=device_add)

    # Parser for "experiment"
    parser_experiment = subparsers.add_parser(
        "experiment", help="Manage and run experiments"
    )
    # Add sub-sub parsers for the experiment module
    exp_subparsers = parser_experiment.add_subparsers(help="experiment submodule help")
    parser_experiment_ls = exp_subparsers.add_parser(
        "ls", help="list experiments and experiment data"
    )
    parser_experiment_ls.add_argument(
        "-n",
        "--name",
        help="list the experiment names",
        action="store_true",
        dest="name",
    )
    parser_experiment_ls.add_argument(
        "-l",
        "--last-run",
        help="list the last time each experiment was run",
        action="store_true",
        dest="last_run",
    )
    parser_experiment_ls.add_argument(
        "-s",
        "--settings",
        help="list the settings for each experiment",
        action="store_true",
        dest="settings",
    )
    parser_experiment_ls.set_defaults(func=experiment_ls)

    parser_experiment_run = exp_subparsers.add_parser("run", help="run an experiment")
    parser_experiment_run.add_argument(
        "name", nargs=1, help="the name of the experiment to be run"
    )
    parser_experiment_run.add_argument(
        "-l",
        "--local",
        help="run the experiment locally using simulated devices",
        action="store_true",
        dest="local",
    )
    parser_experiment_run.add_argument(
        "-o",
        "--output",
        help="specify a location for performance logging output",
        nargs="?",
        dest="output",
    )
    parser_experiment_run.add_argument(
        "-p",
        "--preset",
        help="use a preset runtime config for this launch",
        nargs="?",
        dest="preset",
    )
    parser_experiment_run.set_defaults(func=experiment_run)

    parser_experiment_add = exp_subparsers.add_parser(
        "add", help="add a new experiment"
    )
    parser_experiment_add.add_argument(
        "name", nargs=1, help="the name of the experiment to create"
    )
    parser_experiment_add.set_defaults(func=experiment_add)

    # Parser for 'setup'
    parser_setup = subparsers.add_parser("setup", help="Guided initial setup")

    # Add sub-sub parsers for the setup module
    setup_subparsers = parser_setup.add_subparsers(title="SETUP MODULE COMMANDS")
    parser_setup_controller = setup_subparsers.add_parser(
        "controller", help="Set up the controller"
    )
    parser_setup_controller.add_argument(
        "-r",
        "--reset",
        action="store_true",
        dest="reset",
        help="reset the controller configuration",
    )
    parser_setup_controller.add_argument(
        "-o",
        "--overwrite",
        nargs="+",
        dest="overwrite",
        help="controllerParam=overwriteValue",
    )
    parser_setup_controller.set_defaults(func=setup_controller)

    parser_setup_device = setup_subparsers.add_parser("device", help="Set up a device")
    # TODO: add arguments for device setup
    parser_setup_device.set_defaults(func=setup_device)

    parser_setup_experiment = setup_subparsers.add_parser(
        "experiment", help="Set up an experiment"
    )
    # TODO: add arguments for experiment setup
    parser_setup_experiment.set_defaults(func=setup_experiment)

    args = parser.parse_args()
    if "func" in args:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    # Run the main function
    main()
