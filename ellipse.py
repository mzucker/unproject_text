#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Functions for representing ellipses using various
parameterizations, and converting between them. There are three
parameterizations implemented by this module:

Geometric parameters:
---------------------

The geometric parameters are

  (x₀, y₀, a, b, θ)

The most simple parameterization of an ellipse is by its center point
(x0, y0), its semimajor and semiminor axes a and b, and its rotation
angle θ.

Conic:
------

This parameterization consists of six parameters A-F which establish
the implicit equation for a general conic:

  AX² + BXY + CY² + DX + EY + F = 0

Note that this equation may not represent only ellipses (it also
includes hyperbolas and parabolas).

Since multiplying the entire equation by any non-zero constant results
in the same ellipse, the six parameters are only described up to
scale, yielding five degrees of freedom. We can determine a canonical
scale factor k to scale this equation by such that

  A = a²(sin θ)² + b²(cos θ)²
  B = 2(b² - a²) sin θ cos θ
  C = a²(cos θ)² + b²(sin θ)²
  D = -2Ax₀ - By₀
  E = -Bx₀ - 2Cy₀
  F = Ax₀² + Bx₀y₀ + Cy₀² - a²b²

...in terms of the geometric parameters (x₀, y₀, a, b, θ).

Shape moments:
--------------

The shape moment parameters are

 (m₀₀, m₁₀, m₀₁, mu₂₀, mu₁₁, mu₀₂)

An ellipse may be completely specified by its shape moments up to
order 2. These include the area m₀₀, area-weighted center (m₁₀, m₀₁),
and the three second order central moments (mu₂₀, mu₁₁, mu₀₂).

'''

# pylint: disable=C0103
# pylint: disable=R0914
# pylint: disable=E1101

from __future__ import print_function

import numpy

def _params_str(names, params):

    '''Helper function for printing out the various parameters.'''

    return '({})'.format(', '.join('{}: {:g}'.format(n, p)
                                   for (n, p) in zip(names, params)))

######################################################################

GPARAMS_NAMES = ('x0', 'y0', 'a', 'b', 'theta')
GPARAMS_DISPLAY_NAMES = ('x₀', 'y₀', 'a', 'b', 'θ')

def gparams_str(gparams):
    '''Convert geometric parameters to nice printable string.'''
    return _params_str(GPARAMS_DISPLAY_NAMES, gparams)

def gparams_evaluate(gparams, phi):

    '''Evaluate the parametric formula for an ellipse at each angle
specified in the phi array. Returns two arrays x and y of the same
size as phi.

    '''

    x0, y0, a, b, theta = tuple(gparams)

    c = numpy.cos(theta)
    s = numpy.sin(theta)

    cp = numpy.cos(phi)
    sp = numpy.sin(phi)

    x = a*cp*c - b*sp*s + x0
    y = a*cp*s + b*sp*c + y0

    return x, y

def gparams_from_conic(conic):

    '''Convert the given conic parameters to geometric ellipse parameters.'''

    k, ab = conic_scale(conic)

    if numpy.isinf(ab):
        return None

    A, B, C, D, E, F = tuple(conic)

    T = B**2 - 4*A*C
    
    x0 = (2*C*D - B*E)/T
    y0 = (2*A*E - B*D)/T

    S = A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C)*F
    U = numpy.sqrt((A - C)**2 + B**2)

    a = -numpy.sqrt(2*S*(A+C+U))/T
    b = -numpy.sqrt(2*S*(A+C-U))/T

    theta = numpy.arctan2(C-A-U, B)
    
    return numpy.array((x0, y0, a, b, theta))

def _gparams_sincos_from_moments(m):

    '''Convert from moments to canonical parameters, except postpone the
    final arctan until later. Formulas determined largely by trial and
    error.

    '''

    m00, m10, m01, mu20, mu11, mu02 = tuple(m)

    x0 = m10 / m00
    y0 = m01 / m00

    A = 4*mu02/m00
    B = -8*mu11/m00
    C = 4*mu20/m00

    U = numpy.sqrt((A - C)**2 + B**2)
    T = B**2 - 4*A*C
    S = 1.0

    a = -numpy.sqrt(2*S*(A+C+U))/T
    b = -numpy.sqrt(2*S*(A+C-U))/T

    # we want a * b * pi = m00
    #
    # so if we are off by some factor, we should scale a and b by this factor
    #
    # we need to fix things up somehow because moments have 6 DOF and
    # ellipse has only 5.
    area = numpy.pi * a * b
    scl = numpy.sqrt(m00 / area)
    a *= scl
    b *= scl

    sincos = numpy.array([C-A-U, B])
    sincos /= numpy.linalg.norm(sincos)

    s, c = sincos

    return numpy.array((x0, y0, a, b, s, c))

def gparams_from_moments(m):

    '''Convert the given moment parameters to geometric ellipse parameters.
    Formula derived through trial and error.'''

    x0, y0, a, b, s, c = _gparams_sincos_from_moments(m)
    
    theta = numpy.arctan2(s, c)

    return numpy.array((x0, y0, a, b, theta))

######################################################################

CONIC_NAMES = ('A', 'B', 'C', 'D', 'E', 'F')
CONIC_DISPLAY_NAMES = ('A', 'B', 'C', 'D', 'E', 'F')

def conic_str(conic):

    '''Convert conic parameters to nice printable string.'''
    return _params_str(CONIC_DISPLAY_NAMES, conic)

def conic_scale(conic):

    '''Returns a pair (k, ab) for the given conic parameters, where k is
the scale factor to divide all parameters by in order to normalize
them, and ab is the product of the semimajor and semiminor axis
(i.e. the ellipse's area, divided by pi). If the conic does not
describe an ellipse, then this returns infinity, infinity.

    '''

    A, B, C, D, E, F = tuple(conic)

    T = 4*A*C - B*B

    if T < 0.0:
        return numpy.inf, numpy.inf

    S = A*E**2 + B**2*F + C*D**2 - B*D*E - 4*A*C*F

    if not S:
        return numpy.inf, numpy.inf


    k = 0.25*T**2/S
    ab = 2.0*S/(T*numpy.sqrt(T))

    return k, ab

def conic_from_points(x, y):

    '''Fits conic pararameters using homogeneous least squares. The
resulting fit is unlikely to be numerically robust when the x/y
coordinates given are far from the [-1,1] interval.'''

    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    M = numpy.hstack((x**2, x*y, y**2, x, y, numpy.ones_like(x)))

    _, _, v = numpy.linalg.svd(M)

    return v[5, :].copy()

def conic_transform(conic, H):

    '''Returns the parameters of a conic after being transformed through a
3x3 homography H. This is straightforward since a conic can be
represented as a symmetric matrix (see
https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections).

    '''

    A, B, C, D, E, F = tuple(conic)

    M = numpy.array([[A, 0.5*B, 0.5*D],
                     [0.5*B, C, 0.5*E],
                     [0.5*D, 0.5*E, F]])

    Hinv = numpy.linalg.inv(H)

    M = numpy.dot(Hinv.T, numpy.dot(M, Hinv))

    A = M[0, 0]
    B = M[0, 1]*2
    C = M[1, 1]
    D = M[0, 2]*2
    E = M[1, 2]*2
    F = M[2, 2]

    return numpy.array((A, B, C, D, E, F))

def _conic_from_gparams_sincos(gparams_sincos):

    x0, y0, a, b, s, c = gparams_sincos
    
    A = a**2 * s**2 + b**2 * c**2
    B = 2*(b**2 - a**2) * s * c
    C = a**2 * c**2 + b**2 * s**2
    D = -2*A*x0 - B*y0
    E = -B*x0 - 2*C*y0
    F = A*x0**2 + B*x0*y0 + C*y0**2 - a**2*b**2

    return numpy.array((A, B, C, D, E, F))

def conic_from_gparams(gparams):

    '''Convert geometric parameters to conic parameters. Formulas from
https://en.wikipedia.org/wiki/Ellipse#General_ellipse.

    '''

    x0, y0, a, b, theta = tuple(gparams)
    c = numpy.cos(theta)
    s = numpy.sin(theta)

    return _conic_from_gparams_sincos((x0, y0, a, b, s, c))

def conic_from_moments(moments):

    g = _gparams_sincos_from_moments(moments)
    
    return _conic_from_gparams_sincos(g)

######################################################################

MOMENTS_NAMES = ('m00', 'm10', 'm01', 'mu20', 'mu11', 'mu02')
MOMENTS_DISPLAY_NAMES = ('m₀₀', 'm₁₀', 'm₀₁', 'mu₂₀', 'mu₁₁', 'mu₀₂')

def moments_from_dict(m):

    '''Create shape moments tuple from a dictionary (i.e. returned by cv2.moments).'''
    return numpy.array([m[n] for n in MOMENTS_NAMES])

def moments_str(m):
    '''Convert shape moments to nice printable string.'''
    return _params_str(MOMENTS_DISPLAY_NAMES, m)


def moments_from_gparams(gparams):

    '''Create shape moments from geometric parameters.'''
    x0, y0, a, b, theta = tuple(gparams)
    c = numpy.cos(theta)
    s = numpy.sin(theta)

    m00 = a*b*numpy.pi
    m10 = x0 * m00
    m01 = y0 * m00

    mu20 = (a**2 * c**2 + b**2 * s**2) * m00 * 0.25
    mu11 = -(b**2-a**2) * s * c * m00 * 0.25
    mu02 = (a**2 * s**2 + b**2 * c**2) * m00 * 0.25

    return numpy.array((m00, m10, m01, mu20, mu11, mu02))

def moments_from_conic(scaled_conic):

    '''Create shape moments from conic parameters.'''

    k, ab = conic_scale(scaled_conic)

    if numpy.isinf(ab):
        return None

    conic = numpy.array(scaled_conic)/k

    A, B, C, D, E, _ = tuple(conic)

    x0 = (B*E - 2*C*D)/(4*A*C - B**2)
    y0 = (-2*A*E + B*D)/(4*A*C - B**2)

    m00 = numpy.pi*ab
    m10 = x0*m00
    m01 = y0*m00

    mu20 = 0.25*C*m00
    mu11 = -0.125*B*m00
    mu02 = 0.25*A*m00

    return numpy.array((m00, m10, m01, mu20, mu11, mu02))


######################################################################

def _perspective_transform(pts, H):

    '''Used for testing only.'''

    assert len(pts.shape) == 3
    assert pts.shape[1:] == (1, 2)

    pts = numpy.hstack((pts.reshape((-1, 2)),
                        numpy.ones((len(pts), 1), dtype=pts.dtype)))

    pts = numpy.dot(pts, H.T)

    pts = pts[:, :2] / pts[:, 2].reshape((-1, 1))

    return pts.reshape((-1, 1, 2))

def _test_moments():

    # so I just realized that moments have actually 6 DOF but all
    # ellipse parameterizations have 5, therefore information is lost
    # when going back and forth.
    
    m = numpy.array([59495.5, 5.9232e+07, 1.84847e+07, 3.34079e+08, -1.94055e+08, 3.74633e+08])
    gp = gparams_from_moments(m)
    
    m2 = moments_from_gparams(gp)
    gp2 = gparams_from_moments(m2)

    c = conic_from_moments(m)
    m3 = moments_from_conic(c)

    assert numpy.allclose(gp, gp2)
    assert numpy.allclose(m2, m3)

    print('here is the first thing:')
    print('  {}'.format(moments_str(m)))
    print()
    print('the rest should all be equal pairs:')
    print('  {}'.format(moments_str(m2)))
    print('  {}'.format(moments_str(m3)))
    print()
    print('  {}'.format(gparams_str(gp)))
    print('  {}'.format(gparams_str(gp2)))
    print()


def _test_ellipse():

    print('testing moments badness')
    _test_moments()
    print('pass')

    # test that we can go from conic to geometric and back
    x0 = 450
    y0 = 320
    a = 300
    b = 200
    theta = -0.25

    gparams = numpy.array((x0, y0, a, b, theta))

    conic = conic_from_gparams(gparams)
    k, ab = conic_scale(conic)

    # ensure conic created from geometric params has trivial scale
    assert numpy.allclose((k, ab), (1.0, a*b))

    # evaluate parametric curve at different angles phi
    phi = numpy.linspace(0, 2*numpy.pi, 1001).reshape((-1, 1))
    x, y = gparams_evaluate(gparams, phi)

    # evaluate implicit conic formula at x,y pairs
    M = numpy.hstack((x**2, x*y, y**2, x, y, numpy.ones_like(x)))
    implicit_output = numpy.dot(M, conic)
    implicit_max = numpy.abs(implicit_output).max()

    # ensure implicit evaluates near 0 everywhere
    print('max item from implicit: {} (should be close to 0)'.format(implicit_max))
    print()
    
    assert implicit_max < 1e-5

    # ensure that scaled_conic has the scale we expect
    k = 1e-3
    scaled_conic = conic*k

    k2, ab2 = conic_scale(scaled_conic)

    print('these should all be equal:')
    print()
    print('  k  =', k)
    print('  k2 =', k2)
    assert numpy.allclose((k2, ab2), (k, a*b))
    print()

    # convert the scaled conic back to geometric parameters
    gparams2 = gparams_from_conic(scaled_conic)

    print('  gparams  =', gparams_str(gparams))

    # ensure that converting back from scaled conic to geometric params is correct
    print('  gparams2 =', gparams_str(gparams2))
    assert numpy.allclose(gparams, gparams2)

    # convert original geometric parameters to moments
    m = moments_from_gparams(gparams)
    # ...and back
    gparams3 = gparams_from_moments(m)

    # ensure that converting back from moments to geometric params is correct
    print('  gparams3 =', gparams_str(gparams3))
    print()
    assert numpy.allclose(gparams, gparams3)

    # convert moments parameterization to conic
    conic2 = conic_from_moments(m)

    # ensure that converting from moments to conics is correct
    print('  conic  =', conic_str(conic))
    print('  conic2 =', conic_str(conic2))
    assert numpy.allclose(conic, conic2)

    # create conic from homogeneous least squares fit of points
    skip = len(x) / 10
    conic3 = conic_from_points(x[::skip], y[::skip])

    # ensure that it has non-infinite area
    k3, ab3 = conic_scale(conic3)
    assert not numpy.isinf(ab3)

    # normalize
    conic3 /= k3

    # ensure that conic from HLS fit is same as other 2
    print('  conic3 =', conic_str(conic3))
    print()
    assert numpy.allclose(conic, conic3)

    # convert from conic to moments
    m2 = moments_from_conic(scaled_conic)

    print('  m  =', moments_str(m))

    # ensure that conics->moments yields the same result as geometric
    # params -> moments.
    print('  m2 =', moments_str(m2))
    assert numpy.allclose(m, m2)

    from moments_from_contour import moments_from_contour

    # create moments from contour
    pts = numpy.hstack((x, y)).reshape((-1, 1, 2))
    m3 = moments_from_contour(pts)

    # ensure that moments from contour is reasonably close to moments
    # from geometric params.
    print('  m3 =', moments_str(m3))
    print()
    assert numpy.allclose(m3, m, 1e-4, 1e-4)

    # create a homography H to map the ellipse through
    hx = 0.001
    hy = 0.0015

    H = numpy.array([
        [1, -0.2, 0],
        [0, 0.7, 0],
        [hx, hy, 1]])

    T = numpy.array([
        [1, 0, 400],
        [0, 1, 300],
        [0, 0, 1]])

    H = numpy.dot(T, numpy.dot(H, numpy.linalg.inv(T)))

    # transform the original points thru H
    Hpts = _perspective_transform(pts, H)

    # transform the conic parameters directly thru H
    Hconic = conic_transform(conic, H)

    # get the HLS fit of the conic corresponding to the transformed points
    Hconic2 = conic_from_points(Hpts[::skip, :, 0], Hpts[::skip, :, 1])

    # normalize the two conics
    Hk, Hab = conic_scale(Hconic)
    Hk2, Hab2 = conic_scale(Hconic2)
    assert not numpy.isinf(Hab) and not numpy.isinf(Hab2)

    Hconic /= Hk
    Hconic2 /= Hk2

    # ensure that the two conics are equal
    print('  Hconic  =', conic_str(Hconic))
    print('  Hconic2 =', conic_str(Hconic2))
    print()
    assert numpy.allclose(Hconic, Hconic2)

    # get the moments from Hconic
    Hm = moments_from_conic(Hconic)

    # get the moments from the transformed points
    Hm2 = moments_from_contour(Hpts)

    # ensure that the two moments are close enough
    print('  Hm  =', moments_str(Hm))
    print('  Hm2 =', moments_str(Hm2))
    print()
    assert numpy.allclose(Hm, Hm2, 1e-4, 1e-4)

    # tests complete, now visualize
    print('all tests passed!')

    try:
        import cv2
        print('visualizing results...')
    except ImportError:
        import sys
        print('not visualizing results since module cv2 not found')
        sys.exit(0)

    shift = 3
    pow2 = 2**shift

    p0 = numpy.array([x0, y0], dtype=numpy.float32)
    p1 = _perspective_transform(p0.reshape((-1, 1, 2)), H).flatten()

    Hgparams = gparams_from_conic(Hconic)
    Hp0 = Hgparams[:2]

    skip = len(pts)/100

    display = numpy.zeros((600, 800, 3), numpy.uint8)

    def _asint(x, as_tuple=True):
        x = x*pow2 + 0.5
        x = x.astype(int)
        if as_tuple:
            return tuple(x)
        else:
            return x

    for (a, b) in zip(pts.reshape((-1, 2))[::skip],
                      Hpts.reshape((-1, 2))[::skip]):

        cv2.line(display, _asint(a), _asint(b),
                 (255, 0, 255), 1, cv2.LINE_AA, shift)

    cv2.polylines(display, [_asint(pts, False)], True,
                  (0, 255, 0), 1, cv2.LINE_AA, shift)

    cv2.polylines(display, [_asint(Hpts, False)], True,
                  (0, 0, 255), 1, cv2.LINE_AA, shift)

    r = 3.0

    cv2.circle(display, _asint(p0), int(r*pow2+0.5),
               (0, 255, 0), 1, cv2.LINE_AA, shift)

    cv2.circle(display, _asint(p1), int(r*pow2+0.5),
               (255, 0, 255), 1, cv2.LINE_AA, shift)

    cv2.circle(display, _asint(Hp0), int(r*pow2+0.5),
               (0, 0, 255), 1, cv2.LINE_AA, shift)

    cv2.imshow('win', display)

    print('click in the display window & hit any key to quit.')

    while cv2.waitKey(5) < 0:
        pass

if __name__ == '__main__':

    _test_ellipse()
