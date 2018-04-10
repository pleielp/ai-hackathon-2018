import argparse

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    config = args.parse_args()

    print(args)
    print(config)
    print(config.mode)