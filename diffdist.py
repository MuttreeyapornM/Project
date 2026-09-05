import sys, numpy as np, cv2
sys.path.insert(0,".")
import bev_processor as ORIG, bev_processor_fixed as FIXED
CAMS=["front","left","rear","right"]
SRC={c:f"Dataset/Img_distortion_Testing/{c.capitalize()}.jpg" for c in CAMS}
paths={c:SRC[c] for c in CAMS}
o=ORIG.BEVProcessor(paths,img_car=None); f=FIXED.BEVProcessor(paths,img_car=None)
print(f"{'cam':6s} {'>0':>7s} {'>2':>7s} {'>5':>7s} {'>10':>7s} {'>30':>7s} {'>100':>6s}   of valid px")
for c in CAMS:
    im=cv2.imread(SRC[c])
    if im.shape[:2]!=(720,1280): im=cv2.resize(im,(1280,720))
    _,ow=o.process_image(im,c); _,fw=f.process_image(im,c)
    d=np.abs(ow.astype(np.int16)-fw.astype(np.int16)).max(axis=2)
    valid=(ow.sum(axis=2)>0)|(fw.sum(axis=2)>0)
    n=valid.sum(); dv=d[valid]
    pct=lambda t: 100.0*(dv>t).sum()/n
    print(f"{c:6s} {pct(0):6.2f}% {pct(2):6.2f}% {pct(5):6.2f}% {pct(10):6.2f}% {pct(30):6.2f}% {pct(100):5.2f}%   n={n}")
    # are the big differences on edges?
    g=cv2.cvtColor(ow,cv2.COLOR_BGR2GRAY)
    edge=cv2.dilate(cv2.Canny(g,50,150),np.ones((3,3),np.uint8))>0
    big=(d>30)&valid
    if big.sum():
        print(f"       of the {big.sum()} px differing >30, {100.0*(big&edge).sum()/big.sum():.1f}% lie on a Canny edge")
