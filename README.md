# AI bot for "Nine Men's Morris" Board Game

Try NMM web app: [link](http://nmm.alexandros-kalogerakis.com/)

> app is still in development, may encounter errors

## Approach 1: *Minimax/Alpha Beta Pruning - Neural Networks*

### Training 

```
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
$ pip install -r requirements.txt
$ python3 main.py
```