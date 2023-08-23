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
3. Download the dataset from https://doi.org/10.57760/sciencedb.01684 into data/dataset/

4. To train the model, run
  ```sh
  cd ./train
  python run_me.py
  ```
  
## Data Avilibility

* The datasets generated
and analysed during the current study are publicly available in the Science Data Bank repository at
https://doi.org/10.57760/sciencedb.01684


## Reference

The model is constructed base on P-NET.
https://github.com/marakeby/pnet_prostate_paper

## License

Distributed under the GPL-2.0 License. See `LICENSE` for more information
