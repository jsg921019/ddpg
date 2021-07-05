import cv2
import numpy as np

# block size
a, b = 105, 10

# blank map
map = np.full((6*a + 2*b, 6*a + 2*b, 3), 255, dtype=np.uint8)

# reward tiles
# map[b:b+a, b + int(a*4.5):b + a*6] -= np.array([0,128,0], dtype=np.uint8)
# map[b+a:b+2*a, b + int(a*4.5):b + a*6] -= np.array([0,128,0], dtype=np.uint8)
# map[b+2*a:b+3*a, b + int(a*4.5):b + a*6] -= np.array([0,128,0], dtype=np.uint8)
# map[b+3*a:b+4*a, b + int(a*4.5):b + a*6] -= np.array([0,128,0], dtype=np.uint8)
# map[b+4*a:-b, b + int(a*4.5):b + a*6] -= np.array([0,128,0], dtype=np.uint8)

# map[b+4*a:-b, b + int(a*3):b + int(a*4.5)] -= np.array([0,128,0], dtype=np.uint8)
# map[b+3*a:b+4*a, b + int(a*3):b + int(a*4.5)] -= np.array([0,128,0], dtype=np.uint8)
# map[b+2*a:b+3*a, b + int(a*3):b + int(a*4.5)] -= np.array([0,128,0], dtype=np.uint8)
# map[b+0*a:b+2*a, b + int(a*3):b + int(a*4.5)] -= np.array([0,128,0], dtype=np.uint8)

# map[b+0*a:b+2*a, b + int(a*1.5):b + int(a*3)] -= np.array([0,128,0], dtype=np.uint8)
# map[b+2*a:b+3*a, b + int(a*1.5):b + int(a*3)] -= np.array([0,128,0], dtype=np.uint8)
# map[b+3*a:b+4*a, b + int(a*1.5):b + int(a*3)] -= np.array([0,128,0], dtype=np.uint8)
# map[b+4*a:b+6*a, b + int(a*1.5):b + int(a*3)] -= np.array([0,128,0], dtype=np.uint8)

# map[b+4*a:b+6*a, b + int(a*0):b + int(a*1.5)] -= np.array([0,128,0], dtype=np.uint8)
# map[b+3*a:b+4*a, b + int(a*0):b + int(a*1.5)] -= np.array([0,128,0], dtype=np.uint8)
# map[b+2*a:b+3*a, b + int(a*0):b + int(a*1.5)] -= np.array([0,128,0], dtype=np.uint8)
# map[b+0*a:b+2*a, b + int(a*0):b + int(a*1.5)] -= np.array([0,128,0], dtype=np.uint8)

# borders
map[:b] = 0
map[-b:] = 0
map[:,:b] = 0
map[:,-b:] = 0

# split
map[b:b+a*4, b+a+(a-b)//2:b+a+(a+b)//2] = 0
map[b:b+a*4, b+4*a+(a-b)//2:b+4*a+(a+b)//2] = 0
map[-b-a*4:-b, b+3*a-b//2:b+3*a+b//2] = 0

# reward indices
tiles = [[b, b + a, b + int(a*4.5), b + a*6],
         [b+a, b+2*a, b + int(a*4.5), b + a*6],
         [b+2*a, b+3*a, b + int(a*4.5), b + a*6],
         [b+3*a, b+4*a, b + int(a*4.5), b + a*6],
         [b+4*a, b+6*a, b + int(a*4.5), b + a*6],
         [b+4*a, b+6*a, b + int(a*3), b + int(a*4.5)],
         [b+3*a, b+4*a, b + int(a*3), b + int(a*4.5)],
         [b+2*a,b+3*a, b + int(a*3), b + int(a*4.5)],
         [b+0*a, b+2*a, b + int(a*3), b + int(a*4.5)],
         [b+0*a, b+2*a, b + int(a*1.5), b + int(a*3)],
         [b+2*a, b+3*a, b + int(a*1.5), b + int(a*3)],
         [b+3*a, b+4*a, b + int(a*1.5),b + int(a*3)],
         [b+4*a, b+6*a, b + int(a*1.5), b + int(a*3)],
         [b+4*a, b+6*a, b + int(a*0), b + int(a*1.5)],
         [b+3*a, b+4*a, b + int(a*0), b + int(a*1.5)],
         [b+2*a, b+3*a, b + int(a*0), b + int(a*1.5)],
         [b+0*a, b+2*a, b + int(a*0), b + int(a*1.5)]]

print(tiles[::-1])

for i1, i2, j1, j2 in tiles:
    map_tmp = np.full((6*a + 2*b, 6*a + 2*b, 3), 255, dtype=np.uint8)
    map_tmp[i1:i2, j1:j2] = [255,128,255]
    print(i1,i2, j1, j2)
    map_tmp[:b] = 0
    map_tmp[-b:] = 0
    map_tmp[:,:b] = 0
    map_tmp[:,-b:] = 0
    map_tmp[b:b+a*4, b+a+(a-b)//2:b+a+(a+b)//2] = 0
    map_tmp[b:b+a*4, b+4*a+(a-b)//2:b+4*a+(a+b)//2] = 0
    map_tmp[-b-a*4:-b, b+3*a-b//2:b+3*a+b//2] = 0
    cv2.imshow('map', map_tmp)
    inp = cv2.waitKey(1000)
    if inp != 255:
        break

#cv2.imwrite('map.png', map)

cv2.destroyAllWindows()