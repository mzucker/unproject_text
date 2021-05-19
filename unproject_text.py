#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function

import sys
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import cv2
import ellipse

DEBUG_IMAGES = []

def debug_show(name, src):

    global DEBUG_IMAGES

    filename = 'debug{:02d}_{}.png'.format(len(DEBUG_IMAGES), name)
    cv2.imwrite(filename, src)

    h, w = src.shape[:2]

    fx = w/1280.0
    fy = h/700.0

    f = 1.0/np.ceil(max(fx, fy))

    if f < 1.0:
        img = cv2.resize(src, (0, 0), None, f, f, cv2.INTER_AREA)
    else:
        img = src.copy()

    DEBUG_IMAGES.append(img)

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

def softmax(x, k=1.0):
    b = x.max()
    return np.log( np.exp(k*(x-b)).sum() ) / k + b

def skewed_widths(contours, H):
    xvals = []
    for c in contours:
        pts = cv2.perspectiveTransform(c, H)
        x = pts[:,:,0]
        xvals.append( x.max() - x.min() )
    xvals = np.array(xvals)
    return softmax(xvals, 0.1)

def centered_warp(u0, v0, a, b):
    return np.dot(translation(u0, v0),
                  np.dot(perspective_warp(a, b),
                         translation(-u0, -v0)))

def warp_containing_points(img, pts, H, border=4, shape_only=False):

    '''
    display = img.copy()
    for pt in pts.reshape((-1,2)).astype(int):
        cv2.circle(display, tuple(pt), 4, (255, 0, 0),
                   -1, cv2.LINE_AA)
    debug_show('warp', display)
    '''
    
    pts2 = cv2.perspectiveTransform(pts, H)
    x0, y0, w, h = cv2.boundingRect(pts2)
    print('got bounding rect', x0, y0, w, h)
    T = translation(-x0+border, -y0+border)
    TH = np.dot(T, H)

    if shape_only:
        return (h+2*border, w+2*border), TH
    else:
        dst = cv2.warpPerspective(img, TH, (w+2*border, h+2*border),
                                  borderMode=cv2.BORDER_REPLICATE)
        return dst, TH 

def conic_area_discrepancy(conics, x, H, opt_results=None):

    areas = []
    
    for conic in conics:
        cx = ellipse.conic_transform(conic, H)
        k, ab = ellipse.conic_scale(cx)
        if np.isinf(ab):
            areas.append(1e20)
        else:
            areas.append(ab)

    areas = np.array(areas)
    
    areas /= areas.mean() # rescale so mean is 1.0
    areas -= 1 # subtract off mean
    
    rval = 0.5*np.dot(areas, areas)

    if opt_results is not None:
        if not opt_results or rval < opt_results[-1][-1]:
            opt_results.append( (x, H, rval) )

    return rval

def threshold(img):
    
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mean = img.mean()
    if mean < 100:
        img = 255-img
        
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 101, 21)

def get_contours(img):

    work = threshold(img)

    debug_show('threshold', work)

    contours, hierarchy = cv2.findContours(work, cv2.RETR_CCOMP,
                                              cv2.CHAIN_APPROX_NONE)

    return contours, hierarchy

def get_conics(img, contours, hierarchy,
               abs_area_cutoff=0.0001, mean_area_cutoff=0.15):

    hierarchy = hierarchy.reshape((-1, 4))

    conics = []
    used_contours = []
    areas = []
    okcontours = []
    allchildren = []
    pts = np.empty((0,1,2), dtype='float32')
    centroid_accum = np.zeros(2)
    total_area = 0.0

    centroids = []

    abs_area_cutoff *= img.shape[0] * img.shape[1]
    print('abs_area_cutoff = ',abs_area_cutoff)

    for i, (c, h) in enumerate(zip(contours, hierarchy.reshape((-1, 4)))):

        next_idx, prev_idx, child_idx, parent_idx = h

        if parent_idx >= 0:
            continue

        m = ellipse.moments_from_dict(cv2.moments(c))

        if m[0] <= abs_area_cutoff:
            continue

        children = []

        while child_idx >= 0:
            child_contour = contours[child_idx]
            cm = cv2.moments(child_contour)
            if cm['m00'] > abs_area_cutoff:
                children.append(child_contour)
                allchildren.append(child_contour)
            child_idx = hierarchy[child_idx][0]

        if children:
            work = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(work, contours, i, (1,1,1), -1)
            cv2.drawContours(work, children, -1, (0,0,0), -1)
            m = ellipse.moments_from_dict(cv2.moments(work, True))

        centroids.append(m[1:3]/m[0])
        centroid_accum += m[1:3]
        total_area += m[0]
        pts = np.vstack((pts, c.astype('float32')))
        conic = ellipse.conic_from_moments(m)
        okcontours.append(c)
        conics.append(conic)
        areas.append(m[0])

    display = img.copy()
    cv2.drawContours(display, okcontours+allchildren,
                     -1, (0, 255, 0),
                     6, cv2.LINE_AA)
    
    debug_show('contours_only', display)

    for c, a in zip(okcontours, areas):

        x, y, w, h = cv2.boundingRect(c)

        
        s = str('{:,d}'.format(int(a)))
        #ctr = (x + w/2 - 15*len(s), y+h/2+10)
        ctr = (x, y+h+20)
        
        cv2.putText(display, s, ctr,
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                    (0, 0, 0), 12, cv2.LINE_AA)

        cv2.putText(display, s, ctr,
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                    (0, 255, 0), 6, cv2.LINE_AA)

    debug_show('contours', display)
        
    areas = np.array(areas)
    amean = areas.mean()

    print('got {} contours with {} small.'.format(
        len(areas), (areas < mean_area_cutoff*amean).sum()))
    
    idx = np.where(areas > mean_area_cutoff*amean)[0]

    conics = np.array(conics)
    conics = conics[idx]
    centroid_accum /= total_area

    display = img.copy()
    for conic in conics:
        x0, y0, a, b, theta = ellipse.gparams_from_conic(conic)
        cv2.ellipse(display, (int(x0), int(y0)), (int(a), int(b)),
                    theta*180/np.pi, 0, 360, (0,0,255), 6, cv2.LINE_AA)

    debug_show('conics', display)

    contours = [okcontours[i].astype('float32') for i in idx]

    if 0:

        centroids = np.array([centroids[i] for i in idx])
        areas = areas[idx]

        def polyfit(x, y):
            coeffs = np.polyfit(x, y, deg=1)
            ypred = np.polyval(coeffs, x)
            ymean = np.mean(y)
            sstot = np.sum((y - ymean)**2)
            ssres = np.sum((y.flatten() - ypred.flatten())**2)
            r2 = 1 - ssres/sstot
            return coeffs, r2

        xfit, xr2 = polyfit(centroids[:,0], areas)
        yfit, yr2 = polyfit(centroids[:,1], areas)

        xlabel = 'X coordinate (r²={:.2f})'.format(xr2)
        ylabel = 'Y coordinate (r²={:.2f})'.format(yr2)

        plt.plot(centroids[:,0], areas, 'b.', zorder=1)
        plt.plot(centroids[:,1], areas, 'r.', zorder=1)
        plt.gca().autoscale(False)
        plt.plot([0, 3000], np.polyval(xfit, [0,3000]), 'b--',
                 zorder=0, label=xlabel)
        plt.plot([0, 3000], np.polyval(yfit, [0,3000]), 'r--',
                 zorder=0, label=ylabel)
        plt.legend(loc='upper right')
        plt.xlabel('X/Y coordinate (px)')
        plt.ylabel('Contour area (px²)')
        plt.savefig('position-vs-area.pdf')



    return conics, contours, centroid_accum

def optimize_conics(conics, p0):

    x0 = np.array([0.0, 0.0])
    
    hfunc = lambda x: centered_warp(p0[0], p0[1], x[0], x[1])

    opt_results = []
    
    f = lambda x: conic_area_discrepancy(conics, x, hfunc(x), opt_results)

    res = scipy.optimize.minimize(f, x0, method='Powell')

    H = hfunc(res.x)

    rects = []

    if 0:
        
        phi = np.linspace(0, 2*np.pi, 16, endpoint=False)
        width, height = 0, 0
        for x, H, fval in opt_results:
            allxy = []
            for conic in conics:
                Hconic = ellipse.conic_transform(conic, H)
                gparams = ellipse.gparams_from_conic(Hconic)
                x, y = ellipse.gparams_evaluate(gparams, phi)
                xy = np.dstack((x.reshape((-1, 1, 1)), y.reshape((-1, 1, 1))))
                allxy.append(xy)
            allxy = np.vstack(tuple(allxy)).astype(np.float32)
            rect = cv2.boundingRect(allxy)
            rects.append(rect)
            x, y, w, h = rect
            width = max(width, w)
            height = max(height, h)
        border = int(0.05 * min(width, height))
        width += border
        height += border
        aspect = float(width)/height
        if aspect < 2.0:
            width = 2*height
        else:
            height = width/2
        
        for i, (rect, (x, H, fval)) in enumerate(zip(rects, opt_results)):
            display = np.zeros((height, width), dtype=np.uint8)
            x, y, w, h = rect
            xoffs = width/2 - (x+w/2)
            yoffs = height/2 - (y+h/2)
            for conic in conics:
                Hconic = ellipse.conic_transform(conic, H)
                x0, y0, a, b, theta = ellipse.gparams_from_conic(Hconic)
                cv2.ellipse(display, (int(x0+xoffs), int(y0+yoffs)), (int(a), int(b)),
                            theta*180/np.pi, 0, 360, (255,255,255), 6, cv2.LINE_AA)
            cv2.putText(display, 'Area discrepancy: {:.3f}'.format(fval),
                        (16, height-24), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                        (255,255,255), 6, cv2.LINE_AA)
            cv2.imwrite('frame{:04d}.png'.format(i), display)

    return H

def orientation_detect(img, contours, H, rho=8.0, ntheta=512):

    # ignore this, just deal with edge-detected text

    pts = np.vstack(tuple(contours))
    
    shape, TH = warp_containing_points(img, pts, H, shape_only=True)

    text_edges = np.zeros(shape, dtype=np.uint8)

    for contour in contours:
        contour = cv2.perspectiveTransform(contour.astype(np.float32), TH)
        cv2.drawContours(text_edges, [contour.astype(int)], 0, (255,255,255))
        
    debug_show('edges', text_edges)

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
                 (best_theta_idx, bin_max), (255,0,0),
                 1, cv2.LINE_AA)
        
        debug_show('histogram', debug_hist)

        p0 = np.array((u0, v0))
        t = np.array((np.cos(theta), np.sin(theta)))

        warped = cv2.warpPerspective(img, TH, (shape[1], shape[0]),
                                     borderMode=cv2.BORDER_REPLICATE)

        
        debug_show('prerotate_noline', warped)

        cv2.line(warped,
                 tuple(map(int, p0 - rho*bin_max*t)),
                 tuple(map(int, p0 + rho*bin_max*t)),
                 (255, 0, 0),
                 6, cv2.LINE_AA)

        debug_show('prerotate', warped)

        warped, _ = warp_containing_points(img, pts, RH)
        debug_show('preskew', warped)
        
    return RH


def skew_detect(img, contours, RH):

    hulls = [cv2.convexHull(c) for c in contours]
    pts = np.vstack(tuple(hulls))

    

    display, TRH = warp_containing_points(img, pts, RH)

    for h in hulls:
        h = cv2.perspectiveTransform(h, TRH).astype(int)
        cv2.drawContours(display, [h], 0, (255, 0, 255), 6, cv2.LINE_AA)

    debug_show('convex_hulls_before', display)
    
    f = lambda x: skewed_widths(contours, np.dot(slant(x), RH))

    res = scipy.optimize.minimize_scalar(f, (-2.0, 0.0, 2.0))

    SRH = np.dot(slant(res.x), RH)
    warped, Hfinal = warp_containing_points(img, pts, SRH)

    display = warped.copy()

    for h in hulls:
        h = cv2.perspectiveTransform(h, Hfinal).astype(int)
        cv2.drawContours(display, [h], 0, (255, 0, 255), 6, cv2.LINE_AA)

    debug_show('convex_hulls_after', display)

    debug_show('final', warped)

    return SRH

def main():

    img = cv2.imread(sys.argv[1])
    debug_show('input', img)

    contours, hierarchy = get_contours(img)

    conics, contours, centroid = get_conics(img, contours, hierarchy)
    H = optimize_conics(conics, centroid)
    RH = orientation_detect(img, contours, H)
    SRH = skew_detect(img, contours, RH)

    for img in DEBUG_IMAGES:
        cv2.imshow('Debug', img)
        while cv2.waitKey(5) < 0:
            pass

if __name__ == '__main__':
    main()

