# Simple Collaborative AI Experiments: tracr

tracr is a batteries-included platform that allows users to quickly design and run collaborative inference and split computing experiments over multiple nodes on their local network. It makes it easy to configure remote devices, deploy custom behavior in the form of RPC services, and run experiments with detailed reports and real-time updates.

![output from the "tracr -h" command](./assets/tracrRunDemo.gif)

## Installation

### Requirements
We try to limit dependencies as much as possible, but there are a few. Please review the following requirements before installing tracr:

#### Operating System
Tracr does not officially support Windows, but it should work on any UNIX-like system, including WSL. It has been tested on Ubuntu 20.04 and 21.04.

#### Docker
Docker is used to manage the deployment of services and experiments. To install Docker, follow the instructions for your operating system [here](https://docs.docker.com/get-docker/). If you are using WSL, you will also need to install Docker Desktop for Windows.

#### open-ssh
open-ssh is used to manage remote devices (*"Participants"*) from the local machine (*"Observer"*). All participants must have open-ssh installed and configured to allow passwordless login from the observer. To install open-ssh, run the following commands:

```bash
# For participants:
sudo apt install openssh-server

# For the observer:
sudo apt install openssh-client
```

Additionally, the observer should have [passwordless login](https://phoenixnap.com/kb/setup-passwordless-ssh) set up for each participant.

### Initial Setup

To get started, clone the repository. (The location of the repository doesn't matter):
```bash
git clone https://github.com/nbovee/RACR_AI.git
```

Next, copy the private key for each participant into `AppData/pkeys`. Using the same private key for all participants is preferred because it makes setup easier, but it's fine to use mulitple keys as well.

Once this is done, the last step is to configure the file used to store device information. In the `AppData` directory, there is a file named `known_devices.yaml.example`, which illustrates the format of the device information file. Copy this file to `known_devices.yaml`:

```bash
# (from the root directory of the repository)
cp AppData/known_devices.yaml.example AppData/known_devices.yaml
```

Then replace the example information with the real information for each participant.

Finally, a nice quality-of-life improvement is to add an alias for the run.sh script to your `.bashrc` file. This will allow you to run the script from anywhere on your system using the command `tracr`. To do this, add the following line to your `.bashrc` file:

```bash
alias tracr="bash /path/to/RACR_AI/run.sh"
```

(If you'd prefer not to do this, you can always run the script using its full path).

Congratulations - you are now ready to get started!

## Usage

### Overview
The tracr source code is split into two main sections: `app_api` and `experiment_design`. The `app_api` directory contains the code that runs the tracr application, while the `experiment_design` directory contains the code that defines the behavior of the nodes that will be deployed to participating machines during experiment runtime. To define custom behavior for new experiments, you will extend the code in `experiment_design`. Generally, you will not need to modify the code in `app_api`.

The `experiment_design` directory is split into a number of subdirectories, each representing a customizable part of an experiment. These subdirectories are structured as Python packages, and generally, the user will add a new file to the package and build subclasses of the base classes already defined in the package to implement custom behavior. The subdirectories are as follows:

1. **datasets**: Implements a base class for datasets that can be used in experiments. This is where you will define custom datasets for your experiments. An `imagenet.py` file is included as an example.
2. **models**: Defines the model wrapper used to record fine-grained information about model performance during experiments. 
3. **partitioners**: Defines the partitioner used to split a model into multiple parts for collaborative inference.
4. **records**: Defines the format of the data collected during experiments, as well as the format of the final report.
5. **services**: The most important subdirectory. Defines the base classes for the RPC services that are deployed to each node during experiment runtime. This is where you will define custom behavior for your experiments.
6. **tasks**: Defines the base classes for the tasks that can be performed during experiments. This is where you will define custom tasks for your experiments.

Once an experiment has been designed, it can be run with a single command.

### Designing Experiments
At the most general level, designing an experiment is a matter of:

1. **Defining the behavior** for each node involved in the experiment, which determines how each type of task is handled
2. **Listing the tasks** that must be performed during the experiment
3. **Specifying other parameters**, such as which nodes are deployed to which physical devices, what format to save the report in, etc.

Listing tasks and specifying other parameters is done by creating an *Experiment Manifest* file, which is a YAML file that contains all of the information needed to run an experiment. Defining node behavior is naturally a bit more involved, and is done by extending the base code in `experiment_design`. Let's detail this part first.

#### Defining Node Behavior
A *node*, in this context, is just an exposed RPC service that can be deployed to a physical device (implemented using the [RPyC](https://rpyc.readthedocs.io/en/latest/) library). Each node is responsible for handling a specific type of task. For example, a node might be responsible for performing the first half of an inference on an image, or for finishing that inference. Nodes are deployed to physical devices during experiment runtime, and are then used to perform the tasks that are specified in the experiment manifest.

The vast majority of node behavior is controlled by extending the `ParticipantService` base class in `experiment_design/services/base.py`. An example of this can be found in the file `basic_split_inference.py` in the same directory, where two new types of nodes (`ClientService` and `EdgeService`) are defined. Looking through the example file, you will notice how little code is necessary to implement your own nodes. The basic process for creating a new type of node is as follows:

1. **Create a new class** that extends `ParticipantService`. This class can be defined in its own Python file, or if it's related to other types of nodes, it can be defined in the same file with them.
2. **Override class attributes** `aliases` and `partners`. The `aliases` attribute is a list of strings. The first should be the name of the node, and the second should be the string "PARTICIPANT". The `partners` attribute is a list of strings, where each string is the name of a node that this node can communicate with.
3. **Override methods associated with each type of task** that this node can perform. The dictionary defining these associations is implemented in the base class as the attribute `self.task_map`. To program the node to handle a certain task in a specific way, simply override the method associated with that task.

#### Listing the Tasks
Once the node behavior has been defined, the next step is to list the tasks that must be performed during the experiment. This is done inside the *Experiment Manifest* file, which is a YAML file that contains all of the information needed to run an experiment. It should live inside of its own directory within `UserData/TestCases`. An example manifest can be found in `UserData/TestCases/AlexnetSplit/alexnetsplit.yaml`. 

The manifest is split into three sections:

1. **participant_types**: Defines a type of participant node by pointing to one of the custom service classes defined in `experiment_design/services`, as well as the model that will be used by that class.
2. **participant_instances**: Defines a specific instance of a participant node by pointing to one of the participant types defined in the previous section, as well as the physical device that the node will be deployed to.
3. **playbook**: Defines the sequence of tasks that will be completed during the experiment by assigning a list of tasks to each participant instance from the previous section. These tasks include the task type, and the arguments that will be passed to the task object itself.

The *playbook* section is the focus of this part of the documentation. When an experiment begins, the *Observer* node will iterate through the playbook, use the task name and params to construct a task object, and then send that task object to the appropriate participant node (for all tasks). The participant node stores these tasks in its inbox, which is a PriorityQueue that sorts tasks in a logical order.

Once all the tasks have been sent and the handshake sequence has finished, the participant node will begin processing tasks from its inbox in order of priority. The priority of a task is determined by the task object itself, and is used to ensure that tasks are processed in the correct order. Different types of tasks will have different parameters that must be included in the playbook; check their constructor arguments in `tasks/tasks.py` to see what is required.

#### Specifying Other Parameters
The last step is to specify other parameters, such as which nodes are deployed to which physical devices, what format to save the report in, etc. This, again, is done by editing the *Experiment Manifest* file. Use the *participant_instances* section of the manifest to map participant types to physical devices. More options for experiment customization will be added here soon.

### Running Experiments
Once an experiment has been designed, it can be run with a single command:
  
```bash
tracr experiment run <YOUR_EXPERIMENT_NAME>  # the manifest filename without the .yaml extension
```

Future versions of tracr will allow you to override certain parameters from the manifest on the command line, but for now, you will need to edit the manifest directly to change these parameters.

Once the experiment concludes, the report will be saved in a timestamped file inside the `UserData/TestResults` directory.
