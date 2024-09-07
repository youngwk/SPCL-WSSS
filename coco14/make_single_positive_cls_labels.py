import argparse
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="single_positive_cls_labels_coco.npy", type=str)
    parser.add_argument("--seed", default=1200, type=int, help='random seed')
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    d = np.load('cls_labels_coco.npy', allow_pickle=True).item()

    for img_name, label in d.items():
        spl = np.zeros_like(label)
        idx_all = np.nonzero(label == 1.0)[0]
        if len(idx_all) != 0:
            rng.shuffle(idx_all)
            spl[idx_all[0]] = 1.0
        d[img_name] = spl


    np.save(args.out, d)