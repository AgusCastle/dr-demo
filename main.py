import argparse
import os
from demo import demo

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--save_path', default='./runs/', type=str, help='Path save GradCam Images', required=True)
    parser.add_argument('--device', type=str, default='0', help='Divice on test models cpu, cuda:0, 0.....', required=True)

    args = parser.parse_args()
    path, device = args.save_path, args.device

    assert os.path.exists(path), 'Not found directory'

    demo(device=device, runs=path)





