#!/bin/bash

check_command_exists() {
  command -v "$1" >/dev/null 2>&1
  return $?
}

# Check for conda
if check_command_exists "mamba"; then
  echo "Mamba is installed -> Setting up environment"
  mamba create -n smartamrX python=3.11 numpy scipy keras tensorflow pandas biopython matplotlib seaborn scikit-learn xgboost pytorch
  mamba activate smartamrX
  python ./module_checker.py

else
  echo "Mamba is not installed -> Checking for conda"
  if check_command_exists "conda"; then
    echo "Conda is installed -> Setting up environement"
    conda create -n smartamrX python=3.11 numpy scipy keras tensorflow pandas biopython matplotlib seaborn scikit-learn xgboost pytorch
    conda activate smartamrX
    python ./module_checker.py

  else
    echo "Neither Mamba nor Conda is installed -> Installing mamba"
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh";
    bash Miniforge3-$(uname)-$(uname -m).sh;

    echo "Setting up environment";
    mamba init;
    mamba create -n smartamrX python=3.11 numpy scipy keras tensorflow pandas biopython matplotlib seaborn scikit-learn xgboost pytorch
    mamba activate smartamrX
    python ./module_checker.py
  fi
fi


# If removal nedded : 
# mamba env remove -n smartamrX -y or  conda env remove -n smartamrX -y