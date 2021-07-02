# AI bot for "Nine Men's Morris" Board Game

Try NMM web app: [link](http://nmm.alexandros-kalogerakis.com/)

> app is still in development, may encounter errors

## Approach 1: *Minimax/Alpha Beta Pruning - Neural Networks*

### Training 

```
$ cd 1.MINIMAX_AB_NN
$ pip install -r requirements.txt
$ python3 training.py [-h] -i INPUT -o OUTPUT [-g] [-t {CNN,RNN] [-e EPOCHS] [-b BATCHSIZE] [-lr LR] [-s] [-m] [-f FILTERS] [-d DEPTH]
```

> optional arguments:  

  **-h**, --help: Show this help message and exit  

  **-i** INPUT, --input INPUT: The input numpy dataset filename

  **-o** OUTPUT, --output OUTPUT: The output filename

  **-g**, --gpu: Flag whether to make use of the GPU or not  

  **-t** {CNN,RNN}, --type {CNN,RNN}: Neural Network type *(default: CNN)*

  **-e** EPOCHS, --epochs EPOCHS: Epochs number *(default: 10)*

  **-b** BATCHSIZE, --batch_size BATCHSIZE: Batch Size *(default: 256)*

  **-lr** LR, --learning_rate LR: Learning Rate *(default: 0.0005)*

  **-s**, --stats: Flag whether to plot training statistics or not

  **-m**, --model: Flag whether to plot model schema or not

  **-f** FILTERS, --filters FILTERS: Number of layers' filters *(default: 32)*

  **-d** DEPTH, --depth DEPTH: Neural Network's depth *(default: 6)*



### Play Game
> locally using command line


```
$ cd 1.MINIMAX_AB_NN
$ pip install -r requirements.txt
$ python3 main.py
```

---
## Approach 2: *Reinforcement Learning - SIMPLE - OpenAI Gym*

### Prerequisites

Ubuntu 20.04, not working on Windows
Install [Docker](https://github.com/davidADSP/SIMPLE/issues) and [Docker Compose](https://docs.docker.com/compose/install/) to make use of the `docker-compose.yml` file

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/DevKokko/AI.git
   cd 2.REINFORCEMENT_LEARNING
   ```
2. Build the image and 'up' the container.
   ```sh
   docker-compose up -d
   ```
3. Choose an environment to install in the container (`tictactoe`, `connect4`, `sushigo`, `geschenkt`, `butterfly`, and `flamme rouge` are currently implemented)
   ```sh
   bash ./scripts/install_env.sh ninemensmorris
   ```
  
---
<!-- Running -->
### Running

#### `test.py` 

This entrypoint allows you to play against a trained AI, pit two AIs against eachother or play against a baseline random model.

For example, try the following command to play against the Nine Mens Morris environment.
   ```sh
   docker-compose exec app python3 test.py -d -g 1 -a human best_model -e ninemensmorris 
   ```

#### `train.py` 

This entrypoint allows you to start training the AI using selfplay PPO. The underlying PPO engine is from the [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/) package.

For example, you can start training the agent to learn how to play NineMensMorris with the following command (The following command will reset all the previous training, deleting any previous progress):
   ```sh
   docker-compose exec app python3 train.py -r -e ninemensmorris 
   ```
   
After 30 or 40 iterations the process should have achieved above the default threshold score of 0.2 and will output a new `best_model.zip` to the `/zoo/sushigo` folder. 

Training runs until you kill the process manually (e.g. with Ctrl-C), so do that now.

You can continue training the agent by dropping the `-r` reset flag from the `train.py` entrypoint arguments - it will just pick up from where it left off.

   ```sh
   docker-compose exec app python3 train.py -e ninemensmorris   
   ```
  
---
<!-- Parallelisation -->
### Parallelisation

The training process can be parallelised using MPI across multiple cores.

For example to run 4 parallel threads that contribute games to the current iteration, you can simply run:

  ```sh
  docker-compose exec app mpirun -np 4 python3 train.py -e ninemensmorris 
  ```
  
---
## Additional Feature: *Image Recognition*

### Prerequisites

Python 3.8.10

#### `test.py` 

Uses data from the folder input/test

```
$ cd Image_Recognition
$ pip install -r requirements.txt
$ python3 test.py
```

---

#### `train.py` 

Uses data from the folder input/train

```
$ cd Image_Recognition
$ pip install -r requirements.txt
$ python3 train.py
```