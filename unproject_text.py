import sys
import numpy as np
import scipy.optimize
import cv2
import ellipse

def debug_show(index, name, src):

    filename = 'debug{}_{}.png'.format(index, name)

    cv2.imwrite(filename, src)

    h, w = src.shape[:2]

    fx = w/1280.0
    fy = h/700.0

    f = 1.0/np.ceil(max(fx, fy))

    if f < 1.0:
        img = cv2.resize(src, (0, 0), None, f, f, cv2.INTER_AREA)
    else:
        img = src.copy()

    cv2.imshow('Display', img)
    while cv2.waitKey(5) < 0: pass

def translation(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=float)

def rotation(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

def perspective_warp(a, b):
    return np.array([[1, 0, 0], [0, 1, 0], [a, b, 1]], dtype=float)

def slant(sx):
    return np.array([[1, sx, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

def skewed_widths(contours, H):
    xvals = []
    for c in contours:
        pts = cv2.perspectiveTransform(c, H)
        x = pts[:,:,0]
        xvals.append( x.max() - x.min() )
    xvals = np.array(xvals)
    return np.sum(xvals)

def centered_warp(u0, v0, a, b):
    return np.dot(translation(u0, v0),
                  np.dot(perspective_warp(a, b),
                         translation(-u0, -v0)))

def warp_containing_points(img, pts, H, border=4):
    pts2 = cv2.perspectiveTransform(pts, H)
    x0, y0, w, h = cv2.boundingRect(pts2)
    T = translation(-x0+border, -y0+border)
    TH = np.dot(T, H)
    dst = cv2.warpPerspective(img, TH, (w+2*border, h+2*border),
                              borderMode=cv2.BORDER_REPLICATE)
    return dst, TH 

def conic_area_discrepancy(conics, H):

    areas = []
    
    for conic in conics:
        cx = ellipse.conic_transform(conic, H)
        k, ab = ellipse.conic_scale(cx)
        if np.isinf(ab):
            return np.inf
        areas.append(ab)

    areas = np.array(areas)
    areas /= areas.mean()
    areas -= 1 # subtract off mean
    
    return 0.5*np.dot(areas, areas)

def threshold(img):
    
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 101, 21)

def get_contours(img):

    work = threshold(img)

    if cv2.__version__[0] == '3':
        _, contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours = cv2.findContours(work, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]

    return contours

def get_conics(img, contours, abs_area_cutoff=10.0, mean_area_cutoff=0.15):

    conics = []
    used_contours = []
    areas = []
    pts = np.empty((0,1,2), dtype='float32')
    centroid = np.zeros(2)
    total_area = 0.0

    for c in contours:
        m = ellipse.moments_from_dict(cv2.moments(c))
        if m[0] > abs_area_cutoff:
            centroid += m[1:3]
            total_area += m[0]
            pts = np.vstack((pts, c.astype('float32')))
            conic = ellipse.conic_from_moments(m)
            conics.append(conic)
            areas.append(m[0])

    areas = np.array(areas)
    amean = areas.mean()

    print 'got {} contours with {} small.'.format(
        len(areas), (areas < mean_area_cutoff*amean).sum())
    
    idx = np.where(areas > mean_area_cutoff*amean)[0]

    conics = np.array(conics)
    conics = conics[idx]
    centroid /= total_area

    display = img.copy()
    for conic in conics:
        x0, y0, a, b, theta = ellipse.gparams_from_conic(conic)
        cv2.ellipse(display, (int(x0), int(y0)), (int(a), int(b)),
                    theta*180/np.pi, 0, 360, (0,0,255))

    debug_show(1, 'conics', display)

    contours = [contours[i].astype('float32') for i in idx]

    return conics, contours, centroid

def optimize_conics(conics, p0):

    x0 = np.array([0.0, 0.0])
    
    hfunc = lambda x: centered_warp(p0[0], p0[1], x[0], x[1])
    
    f = lambda x: conic_area_discrepancy(conics, hfunc(x))
    
    with np.errstate(divide='ignore',invalid='ignore'):
        res = scipy.optimize.minimize(f, x0, method='Powell')

    H = hfunc(res.x)

    return H

def orientation_detect(img, contours, H, rho=8.0, ntheta=256):

    # ignore this, just deal with edge-detected text
    pts = np.vstack(tuple(contours))
    warped, _ = warp_containing_points(img, pts, H)
    
    work = threshold(warped)
    text_edges = cv2.Canny(work, 10, 100)

    # generate a linspace of thetas
    thetas = np.linspace(-0.5*np.pi, 0.5*np.pi, ntheta, endpoint=False)

    # rho is pixels per r bin in polar (theta, r) histogram
    # irho is bins per pixel
    irho = 1.0/rho

    # get height and width
    h, w = text_edges.shape

    # maximum bin index is given by hypotenuse of (w, h) divided by pixels per bin
    bin_max = int(np.ceil(np.hypot(w, h)*irho))

    # initialize zeroed histogram height bin_max and width num theta
    hist = np.zeros((bin_max, ntheta))

    # let u and v be x and y coordinates (respectively) of non-zero
    # pixels in edge map
    v, u = np.mgrid[0:h, 0:w]
    v = v[text_edges.view(bool)]
    u = u[text_edges.view(bool)]

    # get center coordinates
    u0 = w*0.5
    v0 = h*0.5

    # for each i and theta = thetas[i]
    for i, theta in enumerate(thetas):

        # for each nonzero edge pixel, compute bin in r direction from
        # pixel location and cos/sin of theta
        bin_idx =  ( (-(u-u0)*np.sin(theta) # x term
                      + (v-v0)*np.cos(theta))*irho # y term, both
                                                   # divided by pixels
                                                   # per bin
                     + 0.5*bin_max ) # offset for center pixel
        
        assert( bin_idx.min() >= 0 and bin_idx.max() < bin_max )

        # 0.5 is for correct rounding here
        #
        # e.g. np.bincount([1, 1, 0, 3]) = [1, 2, 0, 1]
        # returns count of each integer in the array

        bc = np.bincount((bin_idx + 0.5).astype(int))

        # push this into the histogram
        hist[:len(bc),i] = bc

    # number of zero pixels in each column
    num_zero = (hist == 0).sum(axis=0)

    # find the maximum number of zero pixels
    best_theta_idx = num_zero.argmax()

    # actual detected theta - could just return this now
    theta = thetas[best_theta_idx]

    # compose with previous homography 
    RH = np.dot(rotation(-theta), H)
    
    if 1: # just debug visualization

        debug_hist = (255*hist/hist.max()).astype('uint8')
        debug_hist = cv2.cvtColor(debug_hist, cv2.COLOR_GRAY2RGB)

        cv2.line(debug_hist,
                 (best_theta_idx, 0),
                 (best_theta_idx, bin_max), (255,0,0))
        
        debug_show(2, 'histogram', debug_hist)

        p0 = np.array((u0, v0))
        t = np.array((np.cos(theta), np.sin(theta)))

        cv2.line(warped,
                 tuple(map(int, p0 - rho*bin_max*t)),
                 tuple(map(int, p0 + rho*bin_max*t)),
                 (255, 0, 0))

        debug_show(3, 'prerotate', warped)

        warped, _ = warp_containing_points(img, pts, RH)
        debug_show(4, 'preskew', warped)
        
    return RH


def skew_detect(img, contours, RH):

    pts = np.vstack(tuple([cv2.convexHull(c) for c in contours]))

    f = lambda x: skewed_widths(contours, np.dot(slant(x), RH))

    res = scipy.optimize.minimize_scalar(f, (-2.0, 0.0, 2.0))

    SRH = np.dot(slant(res.x), RH)
    warped, Hfinal = warp_containing_points(img, pts, SRH)

    debug_show(5, 'final', warped)

    return SRH

def main():

    img = cv2.imread(sys.argv[1])
    debug_show(0, 'input', img)

    contours = get_contours(img)

    conics, contours, centroid = get_conics(img, contours)
    H = optimize_conics(conics, centroid)
    RH = orientation_detect(img, contours, H)
    SRH = skew_detect(img, contours, RH)

if __name__ == '__main__':
    main()

