# DQN_tfv2
## Requirements

- tensorflow v2.5
- gym[atari]

## Usage

```
$ python main.py --help

$ python main.py --train
$ python main.py --train_continue [weight_file]
$ python main.py --play [weight_file]

usage: main.py [-h] [--train] [--train_continue TRAIN_CONTINUE] [--play PLAY]

Atari: DQN

optional arguments:
  -h, --help            show this help message and exit
  --train               Train agent with given environment
  --train_continue TRAIN_CONTINUE
                        Keep training agent with given environment
  --play PLAY           Play with a given weight directory
```
