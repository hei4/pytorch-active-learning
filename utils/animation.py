import argparse
from PIL import Image
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Making Animation')

    parser.add_argument(
        '--data', '-d', required=True, type=str,
        choices=['moons', 'circles', 'gaussian', 'blobs'])

    parser.add_argument(
        '--algorithm', '-a', required=True, type=str,
        choices=['least', 'margin', 'ratio', 'entropy', 'montecarlo', 'outlier', 'cluster', 'random'])
    
    args = parser.parse_args()

    path = Path(__file__)
    path = path.parents[1].joinpath('results', args.data, args.algorithm)
    filenames = path.glob('*.png')
    filenames = sorted(filenames, key=lambda x: x.name)

    images = []
    for filename in filenames:
        image = Image.open(filename).quantize(colors=256, method=2)     # method 2: fast octree
        images.append(image)
    
    #gifアニメを出力する
    images[0].save(
        path.joinpath(f'{args.data}_{args.algorithm}.gif'),
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=500,
        loop=0)

if __name__ == '__main__':
    main()