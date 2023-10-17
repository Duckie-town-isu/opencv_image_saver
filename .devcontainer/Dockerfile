FROM ubuntu:20.04

SHELL ["/bin/bash", "-c"]

# ~~~~~~~~~~~~~~~~ inside root user ~~~~~~~~~~~~~~~~~

# Minimal setup
RUN apt-get update \
 && apt-get install -y locales lsb-release
ARG DEBIAN_FRONTEND=noninteractive
RUN dpkg-reconfigure locales
 
# Install ROS Noetic
RUN apt-cache policy | grep universe
RUN apt-get install -y software-properties-common
RUN add-apt-repository universe
RUN apt-get install -y wget
RUN apt update -y && apt install -y curl gnupg2 lsb-release
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list

#install ros
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends ros-noetic-desktop-full python3-argcomplete
RUN apt-get install -y --no-install-recommends python3-rosdep
RUN rosdep init \
 && rosdep fix-permissions \
 && rosdep update


RUN source /opt/ros/noetic/setup.bash \
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN source ~/.bashrc


#install ros dev tools
RUN apt update && apt install -y \
  vim \
  libbullet-dev \
  python3-pip \
  python3-pytest-cov \
  ros-dev-tools \
  python3-rosdep \
  python3-rosinstall \
  python3-rosinstall-generator \
  python3-wstool build-essential \
  python3-tk

RUN pip install \
  argcomplete \
  flake8-blind-except \
  flake8-builtins \
  flake8-class-newline \
  flake8-comprehensions \
  flake8-deprecated \
  flake8-docstrings \
  flake8-import-order \
  flake8-quotes \
  pytest-repeat \
  pytest-rerunfailures \
  pytest

RUN pip3 install torch torchvision torchaudio  

RUN apt install --no-install-recommends -y \
  libasio-dev \
  libtinyxml2-dev

RUN apt install --no-install-recommends -y \
  libcunit1-dev

# install pip libraries
# RUN pip install <name>
RUN pip install shapely

#install ros dev tools
# RUN apt update && apt install -y \

#source necessary things
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

RUN cd ~

