import numpy

def eigh_2x2(x, y, z):

    q = numpy.sqrt(z**2 - 2*x*z + 4*y**2 + x**2)

    w = numpy.array( [
         0.5 * (z + x - q),
         0.5 * (z + x + q) ] )

    V = numpy.array( [
        [ 2*y,     2*y   ],
        [ (z-x-q), (z-x+q) ] ] )
    
    return w, V

######################################################################

def conic_names():
    return ('A', 'B', 'C', 'D', 'E', 'F')

def conic_scale(conic):
    
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

    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    M = numpy.hstack( ( x**2, x*y, y**2, x, y, numpy.ones_like(x) ) )

    u, s, v = numpy.linalg.svd(M)

    return v[5,:].copy()

def conic_transform(conic, H):

    A, B, C, D, E, F = tuple(conic)
    
    M = numpy.array([
        [ A, 0.5*B, 0.5*D ],
        [ 0.5*B, C, 0.5*E ],
        [ 0.5*D, 0.5*E, F ] ])
    
    Hinv = numpy.linalg.inv(H)

    M = numpy.dot(Hinv.T, numpy.dot(M, Hinv))
    
    A = M[0,0]
    B = M[0,1]*2
    C = M[1,1]
    D = M[0,2]*2
    E = M[1,2]*2
    F = M[2,2]

    return numpy.array((A, B, C, D, E, F))

def conic_from_params(params):

    x0, y0, a, b, theta = tuple(params)
    c = numpy.cos(theta)
    s = numpy.sin(theta)

    A = a**2 * s**2 + b**2 * c**2
    B = 2*(b**2 - a**2) * s * c
    C = a**2 * c**2 + b**2 * s**2
    D = -2*A*x0 - B*y0
    E = -B*x0 - 2*C*y0
    F = A*x0**2 + B*x0*y0 + C*y0**2 - a**2*b**2

    return numpy.array((A, B, C, D, E, F))

def conic_from_moments(moments):

    m00, m10, m01, mu20, mu11, mu02 = tuple(moments)

    x0 = m10/m00
    y0 = m01/m00
    
    A = 4*mu02/m00
    B = -8*mu11/m00
    C = 4*mu20/m00

    a2b2 = 0.25*(4*A*C - B*B)

    D = -2*A*x0 - B*y0
    E = -B*x0 - 2*C*y0
    F = A*x0**2 + B*x0*y0 + C*y0**2 - a2b2

    return numpy.array((A, B, C, D, E, F))

######################################################################

def param_names():
    return ('x0', 'y0', 'a', 'b', 'theta')

def params_evaluate(params, phi):

    x0, y0, a, b, theta = tuple(params)
    
    c = numpy.cos(theta)
    s = numpy.sin(theta)

    cp = numpy.cos(phi)
    sp = numpy.sin(phi)

    x = a*cp*c - b*sp*s + x0
    y = a*cp*s + b*sp*c + y0

    return x, y

def params_from_conic(conic):

    k, ab = conic_scale(conic)
    
    if numpy.isinf(ab):
        return None
    
    A, B, C, D, E, F = tuple(conic)

    x0 = (B*E - 2*C*D)/(4*A*C - B**2)
    y0 = (-2*A*E + B*D)/(4*A*C - B**2)

    w, V = eigh_2x2(A, 0.5*B, C)
    
    b, a = tuple(numpy.sqrt(w/k))
    theta = numpy.arctan2(-V[0,1], V[1,1])

    return numpy.array((x0, y0, a, b, theta))

def params_from_moments(m):

    m00, m10, m01, mu20, mu11, mu02 = tuple(m)

    x0 = m10 / m00
    y0 = m01 / m00

    w, V = eigh_2x2(mu20/m00, mu11/m00, mu02/m00)

    b, a = tuple(2.0*numpy.sqrt(w))
    theta = numpy.arctan2(V[0,0], -V[1,0])

    return numpy.array((x0, y0, a, b, theta))

######################################################################

def moments_names():
    return ('m00', 'm10', 'm01', 'mu20', 'mu11', 'mu02')

def moments_from_dict(m):
    return numpy.array( [ m[n] for n in moments_names() ] )

def moments_from_params(params):

    x0, y0, a, b, theta = tuple(params)
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
    
    k, ab = conic_scale(scaled_conic)
    
    if numpy.isinf(ab):
        return None
    
    conic = numpy.array(scaled_conic)/k
    
    A, B, C, D, E, F = tuple(conic)

    x0 = (B*E - 2*C*D)/(4*A*C - B**2)
    y0 = (-2*A*E + B*D)/(4*A*C - B**2)
    
    m00 = numpy.pi*ab
    m10 = x0*m00
    m01 = y0*m00

    mu20 = 0.25*C*m00
    mu11 = -0.125*B*m00
    mu02 = 0.25*A*m00

    return numpy.array((m00, m10, m01, mu20, mu11, mu02))
