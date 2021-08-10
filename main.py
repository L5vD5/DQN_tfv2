import argparse

from Agent import Agent

def args_parse():
    parser = argparse.ArgumentParser(description="Atari: DQN")
    parser.add_argument('--train', action="store_true", help='Train agent with given environment')
    parser.add_argument('--train_continue', help='Keep training agent with given environment')
    parser.add_argument('--play', help="Play with a given weight directory")
    parser.add_argument('--gamma', default=0.99, help="discount factor")
    parser.add_argument('--buffer_size', default=10000, help="How big buffer size be")
    parser.add_argument('--mini_batch_size', default=32, help="How big mini batch size be")
    parser.add_argument('--epochs', default=1000000, help="Number of epochs of interaction (equivalent to number of policy updates) to perform")
    parser.add_argument('--evaluate_every', default=10, help="How often evaluate be")
    parser.add_argument('--steps_per_epoch', default=5000, help="How many steps per epoch")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = args_parse()
    a = Agent(args)
    if args.train:
        print("Start training")
        a.train()
    elif args.play:
        print("Start playing")
        a.play(args.play)
    elif args.train_continue:
        a.train(args.train_continue, 1000000)
