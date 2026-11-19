"""Image-only subpixel junction and outer-edge measurements for the 80 mm ID 0 marker."""
import cv2
import numpy as np


def internal_corners(image,uv):
    """Estimate marker homography from internal grid junctions, excluding its rim."""
    tag=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000),0,80)
    bits=tag[5::10,5::10]>0;grid=[]
    for j in range(1,8):
        for i in range(1,8):
            a,b,c,d=bits[j-1:j+1,i-1:i+1].ravel()
            if sum([int(a),int(b),int(c),int(d)]) in (1,3) or (a==d and b==c and a!=b):grid.append((i,j))
    grid=np.asarray(grid,np.float32);outer=np.array([[0,0],[8,0],[8,8],[0,8]],np.float32)
    H=cv2.getPerspectiveTransform(outer,np.asarray(uv,np.float32))
    initial=cv2.perspectiveTransform(grid[:,None],H)[:,0]
    cell=np.linalg.norm(np.asarray(uv)-np.roll(uv,1,axis=0),axis=1).min()/8
    win=max(2,min(5,int(cell*.30)))
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    refined=cv2.cornerSubPix(gray,initial.copy(),(win,win),(-1,-1),(3,40,.001))
    good=np.linalg.norm(refined-initial,axis=1)<min(3.,cell*.35)
    if good.sum()<12:return np.asarray(uv),{'accepted':False,'reason':'insufficient_internal_corners'}
    H,inliers=cv2.findHomography(grid[good],refined[good],cv2.RANSAC,.6,maxIters=2000,confidence=.999)
    if H is None or inliers.sum()<12:return np.asarray(uv),{'accepted':False,'reason':'internal_homography_rejected'}
    accepted=inliers.ravel().astype(bool)
    H,_=cv2.findHomography(grid[good][accepted],refined[good][accepted],0)
    corners=cv2.perspectiveTransform(outer[:,None],H)[:,0]
    if np.linalg.norm(corners-uv,axis=1).max()>5:return np.asarray(uv),{'accepted':False,'reason':'internal_large_shift'}
    error=cv2.perspectiveTransform(grid[good][accepted,None],H)[:,0]-refined[good][accepted]
    return corners,{'accepted':True,'grid_points':grid[good][accepted].tolist(),'image_points':refined[good][accepted].tolist(),
                   'rms_px':float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),'max_shift_px':float(np.linalg.norm(corners-uv,axis=1).max())}


def edge_corners(image,uv):
    """Fit outer black/white transitions away from corners; no robot/model input."""
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray=cv2.GaussianBlur(gray,(0,0),.6)
    uv=np.asarray(uv,float);lines=[];diagnostics=[]
    for a,b in zip(uv,np.roll(uv,-1,axis=0)):
        tangent=b-a;length=np.linalg.norm(tangent);tangent/=length
        normal=np.array([-tangent[1],tangent[0]])
        fractions=np.linspace(.15,.85,max(25,int(length*.65)))
        center=a+fractions[:,None]*(b-a); offsets=np.linspace(-3.5,3.5,57)
        coords=center[:,None,:]+offsets[None,:,None]*normal
        values=cv2.remap(gray,coords[:,:,0].astype(np.float32),coords[:,:,1].astype(np.float32),cv2.INTER_LINEAR)
        gradient=-np.gradient(values,offsets,axis=1);peaks=gradient[:,2:-2].argmax(axis=1)+2
        row=np.arange(len(center));strength=gradient[row,peaks]
        lo,mid,hi=gradient[row,peaks-1],strength,gradient[row,peaks+1]
        shift=np.clip(.5*(lo-hi)/(lo-2*mid+hi-1e-12),-.5,.5)
        positions=offsets[peaks]+shift*(offsets[1]-offsets[0])
        good=(strength>10)&(abs(positions)<3.2)
        if good.sum()<max(15,len(center)*.5): return uv,{'accepted':False,'reason':'insufficient_edge_contrast'}
        points=(center+positions[:,None]*normal)[good]
        for _ in range(3):
            loc=points.mean(axis=0);_,_,V=np.linalg.svd(points-loc,full_matrices=False);n=V[-1]
            residual=(points-loc)@n;scale=max(.15,1.4826*np.median(abs(residual-np.median(residual))))
            keep=abs(residual)<max(.5,3*scale)
            if keep.sum()<15:return uv,{'accepted':False,'reason':'edge_outliers'}
            points=points[keep]
        loc=points.mean(axis=0);_,_,V=np.linalg.svd(points-loc,full_matrices=False);n=V[-1]
        lines.append((n,float(n@loc)));diagnostics.append(float(np.std((points-loc)@n)))
    refined=[]
    for i in range(4):
        n1,c1=lines[i-1];n2,c2=lines[i];A=np.array([n1,n2])
        if abs(np.linalg.det(A))<.1:return uv,{'accepted':False,'reason':'near_parallel_edges'}
        refined.append(np.linalg.solve(A,[c1,c2]))
    refined=np.asarray(refined)
    if np.linalg.norm(refined-uv,axis=1).max()>4:return uv,{'accepted':False,'reason':'large_corner_shift'}
    return refined,{'accepted':True,'line_residual_std_px':diagnostics,'max_shift_px':float(np.linalg.norm(refined-uv,axis=1).max())}
