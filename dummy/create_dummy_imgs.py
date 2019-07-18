import numpy as np
from PIL import Image

if __name__ == '__main__':
    for i in range(128):
        imarray = np.random.rand(320, 320) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('L')
        im.save('dummy' + str(i) + '.jpeg')
        
