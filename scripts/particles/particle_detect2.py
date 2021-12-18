import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

IMAGE="C:\\Users\\be34gof\\Downloads\\20211206\\PE_226-23h_06.tif"


#def main():
im = Image.open(IMAGE)
imarray = np.array(im)
footer_idx=(imarray.mean(1)[int(imarray.shape[0]/2):]>200).argmax()+int(imarray.shape[0]/2)


oimg = imarray[:footer_idx]
img=oimg - oimg.min()
img=img/img.max()
grayImage = cv2.cvtColor((img*255).astype('uint8'), cv2.COLOR_GRAY2BGR)


circles = cv2.HoughCircles((img*255).astype('uint8'), cv2.HOUGH_GRADIENT, 1, 20,
                           param1=110, param2=20, minRadius=4, maxRadius=30
                           )
plt.imshow(img)

if circles is not None:
    for x, y, r in sorted(circles[0],key=lambda d:d[2]):
        c = plt.Circle((x, y), r, fill=False, lw=2, ec='C1')
        c2 = plt.Circle((x, y), r, fill=True, lw=0, ec='C1')
        plt.gca().add_patch(c)
        plt.gca().add_patch(c2)
plt.show()
plt.close()


contourimage=img.copy()
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        round=img[max(0,x-1):min(x+2,img.shape[0]),max(0,y-1):min(y+2,img.shape[1])]
        contourimage[x,y]=round.max()-round.min()


contourimage[contourimage<0.1]=0
contourimage[contourimage>0]=1
contourimage=contourimage.astype(bool)
plt.imshow(contourimage)
plt.colorbar()
plt.show()
plt.close()


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
is_particle[contourimage]=False
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

particles=[]
while is_particle.max()>0:
    s = np.unravel_index(is_particle.argmax(), nan_dist.shape)
    indices_to_check=[s]
    indices_checked=[]

    part_of_p = np.zeros_like(is_particle)
    ps=-1
    while ps<part_of_p.sum():
        ps=part_of_p.sum()
        ni2c=[]
        for s in indices_to_check:
            if s in indices_checked:
                continue
            indices_checked.append(s)
            if is_particle[s]:
                if s[0]>0:
                    ni2c.append((s[0]-1,s[1]))
                if s[0]<is_particle.shape[0]-1:
                    ni2c.append((s[0]+1,s[1]))
                if s[1]>0:
                    ni2c.append((s[0],s[1]-1))
                if s[1]<is_particle.shape[1]-1:
                    ni2c.append((s[0],s[1]+1))
                part_of_p[s]=True
        indices_to_check.extend(ni2c)
    particles.append(part_of_p)
    is_particle[part_of_p]=0
    plt.imshow(part_of_p)
    plt.show()
    plt.close()


MAXR=50
rads=np.arange(1,MAXR)
plt.imshow(oimg)
circ_base=np.zeros_like(whatershed)+(whatershed.shape[0]*whatershed.shape[0])+(whatershed.shape[1]*whatershed.shape[1])
for p in particles:
    _whatershed=whatershed.copy()
    _whatershed[~p]=0
    cx,cy=np.unravel_index(_whatershed.argmax(), nan_dist.shape)
    circ=circ_base.copy()
    d=[]

    for x in range(max(0,cx-MAXR),min(cx+MAXR+1,whatershed.shape[0])):
        for y in range(max(0,cy-MAXR),min(cy+MAXR+1,whatershed.shape[1])):
            circ[x,y]=(x-cx)**2+(y-cy)**2

    for r in rads:
        mask = circ < r**2
        d.append(whatershed[mask].min())
    d=np.array(d)
    rm=d.argmin()+1
    angle = np.linspace( 0 , 2 * np.pi , 150 )
    x = rm * np.cos( angle ) + cy
    y = rm * np.sin( angle ) + cx
    plt.plot(x,y,"red")


plt.show()
plt.close()
#if __name__ == '__main__':
#    main()

