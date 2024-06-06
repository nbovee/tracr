Inspiration from <https://github.com/gooooloo/grpc-file-transfer>
Info on pita from python grpc <https://news.ycombinator.com/item?id=21873468>
generated code imports must be converted to python 3 formatting, or some edit to the generation command is required
custom install of <https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8_8.4.0.27-1+cuda11.6_amd64.deb>

<https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html>
<https://github.com/microsoft/WSL/issues/4150#issuecomment-504209723>
<https://github.com/dusty-nv/jetson-containers>
<https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-ml>
<https://github.com/pjreddie/darknet/tree/master/src>
<https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch>
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>
<https://github.com/WongKinYiu/ScaledYOLOv4>
<https://paperswithcode.com/method/yolov4>
<>


### Changes introduced in the RPi3 Compatibility Feature Branch
* A few utility scripts have been added to make it easier to interface with the IoT devices (RPi3s, Jetson Nanos) from a PC
  * open_gateway.sh: Allows the IoT devices in the CS Lab LAN to access the internet using your PC as a gateway. Helpful for installing new software or updates.
    * Must connect your PC to the LAN via ethernet and set your ethernet adaptor to use the static IP 192.168.1.200. I'm also pretty sure you need to be using a Debian-based OS.
  * sync_repos.sh: Quickly syncs changes made to the collab-vision repo on your local machine to all IoT devices in the LAN.
    * Must have passwordless (pub/private key) ssh set up for each IoT device. Use ssh-keygen to get this set up quickly.
  * deploy_node.sh: Deploys a containerized "node" for testing as either a remote or client device. Pass option flags to determine which base image to use, depending on the architecture and hardware of the node.
    * There are a few options - check the source code for a longer description comment
    * Also gives you the ability to run any image in "Interactive Terminal" mode so you can use a terminal from inside the running container, test, and poke around for troubleshooting.
* Docker Image tarballs are being saved directly in the repo
  * Makes it a little easier to ensure each device has access to the same images
  * Using docker load command to load images from the tarballs if they are not already present in the local system (this happens in deploy_node.sh)
* The process for building containers with new functionality has changed a bit
  * The base images simply provide the same clean slate (with Python 3.7, torch 1.10, torchvision 0.11, etc.) for multiple architectures (arm64 with cuda, arm64 without cuda, x86 with and without).
  * To specify the additional setup and final execution for a new container, create a new Dockerfile that starts from a base image, sets up what it has to, then executes the python program you want to run.
    * The suffix you choose for the Dockerfile will be used to tag this image for future use, so try to be descriptive.
    * Like before, the Dockerfile will still accept a "branch" argument to implement remote or client functionality in the same way.
* I have begun working on a "local_testing" directory for roughly testing new approaches on a virtual bridge network so we can see how things work without actually having physical access to the LAN in the CS lab.
  * There is a docker-compose.yml file that doesn't work as intended (yet).
* I added a text file named "device_info" to the .gitignore file since it contains semi-private information about device IPs and passwords, but the info has been posted in the RACR discord.
* We are successfully running PyTorch and torchvision on the Raspberry Pi 3 devices inside a docker container.
