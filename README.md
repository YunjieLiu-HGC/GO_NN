# GO_NN

## Description 

* Gene Ontology informed neural network model formicrobe functionality discovery in human diseases.

## Environments
* python 2.7, check environments.yml for needed packages

## Installation

1. Clone the repository 
  ```sh
  git clone https://github.com/YunjieLiu-HGC/GO_NN.git
  ```
2. Create conda environment
  ```sh
  conda env create --name gonn --file=environment.yml
  ```
## Usage

1. Activate the created conda environment
  ```sh
  source activate gonn
  ```
2. Add current directory to PYTHONPATH
  ```sh
  export PYTHONPATH=~/GO_NN:$PYTHONPATH
  ```
3. To train the model, run
  ```sh
  cd ./train
  python run_me.py
  ```
## License

Distributed under the GPL-2.0 License. See `LICENSE` for more information
