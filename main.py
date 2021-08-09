import argparse

from Agent import Agent

def args_parse():
    parser = argparse.ArgumentParser(description="Atari: DQN")
    parser.add_argument('--train', action="store_true", help='Train agent with given environment')
    parser.add_argument('--train_continue', help='Keep training agent with given environment')
    parser.add_argument('--play', help="Play with a given weight directory")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parse()
    a = Agent()
    if args.train:
        print("Start training")
        a.train()
    elif args.play:
        print("Start playing")
        a.play(args.play)
    elif args.train_continue:
        a.train(args.train_continue, 1000000)
