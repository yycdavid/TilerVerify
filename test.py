from PIL import Image
import h5py
import numpy as np

def main():
    with h5py.File('perturbed.mat', 'r') as f:
        for k, v in f.items():
            img_data = np.array(v)
    img_data = np.transpose(np.clip(np.floor(img_data*256),0,255))
    img = Image.fromarray(img_data)
    img = img.convert("L")
    img.save('perturbed.jpg')

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
