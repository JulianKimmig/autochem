import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

IMAGE="C:\\Users\\Julian_Stobbe\\Downloads\\20211206\\PE_226-23h_06.tif"


def main():
    im = Image.open(IMAGE)
    imarray = np.array(im)
    footer_idx=(imarray.mean(1)[int(imarray.shape[0]/2):]>200).argmax()+int(imarray.shape[0]/2)
    img = imarray[:footer_idx]
    img-=img.min()
    img=img/img.max()
    img[(img-np.median(img))<0.001]=np.nan
    nan_dist=np.zeros_like(img)
    nan_dist[~np.isnan(img)]=np.nan

    #xidx,yidx = (np.unravel_index(np.argwhere(np.isnan(nan_dist)), nan_dist.shape))
    nan_idx=np.argwhere(np.isnan(nan_dist))
    presize=nan_idx.size + 1
    while nan_idx.size<presize:
        presize=nan_idx.size
        nnan_dist=nan_dist.copy()
        for x,y in nan_idx:
            #print(x,y)
            round=nan_dist[max(0,x-1):min(x+2,nan_dist.shape[0]),max(0,y-1):min(y+2,nan_dist.shape[1])]
            if not np.all(np.isnan(round)):
                nnan_dist[x,y]=np.nanmin(round)+1

        nan_dist=nnan_dist
        nan_idx = np.argwhere(np.isnan(nan_dist))


    whatershed=nan_dist.copy()
    plt.imshow(whatershed)
    plt.show()
    plt.close()

    is_particle=whatershed>5
    plt.imshow(is_particle)
    plt.show()
    plt.close()

    for i in range(5):
        is_particle_c=is_particle.copy()
        for x in range(is_particle.shape[0]):
            for y in range(is_particle.shape[1]):
                if is_particle[x,y]:
                    round=is_particle[max(0,x-1):min(x+2,is_particle.shape[0]),max(0,y-1):min(y+2,is_particle.shape[1])].astype(int)
                    is_particle_c[x,y]=round.sum()/round.size > 0.6
        is_particle=is_particle_c
        plt.imshow(is_particle)
        plt.show()
        plt.close()

if __name__ == '__main__':
    main()

