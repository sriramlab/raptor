import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--saveas', type=str, required=True)
    parser.add_argument('--d', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    import numpy as np
    np.random.seed(args.seed)

    rp = np.random.randn(args.d, args.k)

    np.save(args.saveas, rp)
