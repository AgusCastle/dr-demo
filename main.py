import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')

    # paths
    parser.add_argument('--models_path', default='./models/', type=str, help='Models paths (Convnext_#### and SNF)')
    parser.add_argument('--src_path', default='./imgs/', type=str, help='Images to predict paths')

    parser.add_argument('--save_path', default='./runs/', type=str, help='Path save GradCam Images')
    # flags
    parser.add_argument('--save', default=True, help='Save all results from your entry images')
    parser.add_argument('--device', type=str, default='0', help='Divice on test models cpu, cuda:0, 0.....')