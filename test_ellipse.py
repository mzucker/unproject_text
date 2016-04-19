import cv2
from ellipse import *


######################################################################
# Things to test:
#
#   closed-form computation of eigenvalues and eigenvectors of 2x2
#   Hermitian matrix.
#
#   params_from_conic(conic_from_params(params)) == params
#   params2
#
#   conic_from_params(params_from_conic(conic)) == conic
#   Don't need to test this since params_from_conic(conic) == params
#
#   moments_from_params(params) == cv2.moments(...)
#   m2
#
#   moments_from_conic(conic) == moments_from_params(params)
#   m3
#   
#   params_from_moments(moments_from_params(params)) == params
#   params3
# 
#   conic_from_moments(moments_from_conic(conic)) == conic
#   conic2
#
#   map ellipse though homography works


for i in range(100):
    theta = numpy.random.random()*2*numpy.pi
    a = numpy.random.random()*9 + 1
    b = numpy.random.random()*9 + 1
    a, b = max(a,b), min(a,b)
    c = numpy.cos(theta)
    s = numpy.sin(theta)
    R = numpy.array([[c, -s], [s, c]])
    A = numpy.dot(R, numpy.dot(numpy.diag([a,b]), R.T))
    assert(numpy.allclose(A[1,0], A[0,1]))
    w, V = eigh_2x2(A[0,0], A[0,1], A[1,1])
    AA = numpy.dot(V, numpy.dot(numpy.diag(w), numpy.linalg.inv(V)))
    assert(numpy.allclose(w, [b, a]))
    assert(numpy.allclose(A, AA))

x0 = 450
y0 = 320
a = 300
b = 200
theta = -0.25

params = numpy.array((x0, y0, a, b, theta))
conic = conic_from_params(params)

k = 1e-3
scaled_conic = conic * k

ab = a*b

phi = numpy.linspace(0, 2*numpy.pi, 1001).reshape((-1,1))

x, y = params_evaluate(params, phi)

M = numpy.hstack((x**2, x*y, y**2, x, y, numpy.ones_like(x)))

foo = numpy.dot(M, conic)
foomax = numpy.abs(foo).max()

print 'should all be 0:\n', foo
print 'max:', foomax 
print
assert(foomax < 1e-3)

k2, ab2 = conic_scale(scaled_conic)
print 'k:', k
print 'k2:', k2
print
assert(numpy.allclose(k2, k))

params2 = params_from_conic(scaled_conic)

m = moments_from_params(params)

params3 = params_from_moments(m)
conic2 = conic_from_moments(m)

skip = len(x) / 10
conic3 = conic_from_points(x[::skip], y[::skip])
k3, ab3 = conic_scale(conic3)
assert(not numpy.isinf(ab3))
conic3 /= k3

print 'params:', tuple(params)
print 'params2:', tuple(params2)
print 'params3:', tuple(params3)
print
assert(numpy.allclose(params, params2))
assert(numpy.allclose(params, params3))

print 'conic:', tuple(conic)
print 'conic2:', tuple(conic2)
print 'conic3:', tuple(conic3)
print
assert(numpy.allclose(conic, conic2))
assert(numpy.allclose(conic, conic3))

m3 = moments_from_conic(scaled_conic)


pts = numpy.hstack((x,y)).astype('float32').reshape((-1,1,2))
m2 = moments_from_dict(cv2.moments(pts, False))

print 'm:', tuple(m)
print 'm2:', tuple(m2)
print 'm3:', tuple(m3)
print
assert(numpy.allclose(m2, m, 1e-4, 1e-4))
assert(numpy.allclose(m, m3))

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

Tinv = numpy.linalg.inv(T)

H = numpy.dot(T, numpy.dot(H, Tinv))


print 'H:\n', H
print


Hpts = cv2.perspectiveTransform(pts, H)

Hconic = conic_transform(conic, H)
Hconic2 = conic_from_points(Hpts[::skip,:,0], Hpts[::skip,:,1])

Hk, Hab = conic_scale(Hconic)
Hk2, Hab2 = conic_scale(Hconic2)

print 'Hconic:', tuple(Hconic/Hk)
print 'Hconic2:', tuple(Hconic2/Hk2)
print
assert(numpy.allclose(Hconic/Hk, Hconic2/Hk2))


Hm = moments_from_conic(Hconic)
Hm2 = moments_from_dict(cv2.moments(Hpts, False))

Hparams = params_from_conic(Hconic)

print 'Hm:', tuple(Hm)
print 'Hm2:', tuple(Hm2)
print
assert(numpy.allclose(Hm, Hm2, 1e-4, 1e-4))

display = numpy.zeros((600, 800, 3), numpy.uint8)

shift = 3
pow2 = 2**shift

p0 = numpy.array([x0, y0], dtype=numpy.float32)
p1 = cv2.perspectiveTransform(p0.reshape((-1, 1, 2)), H).flatten()
Hp0 = Hparams[:2]

skip = len(pts)/100

for (a, b) in zip(pts.reshape((-1,2))[::skip], Hpts.reshape((-1,2))[::skip]):
    cv2.line(display,
             tuple(map(int, (a*pow2+0.5))),
             tuple(map(int, (b*pow2+0.5))),
             (255, 0, 255), 1, cv2.LINE_AA, shift)

cv2.polylines(display, [(pts*pow2+0.5).astype(int)], True,
              (0, 255, 0), 1, cv2.LINE_AA, shift)

cv2.polylines(display, [(Hpts*pow2+0.5).astype(int)], True,
              (0, 0, 255), 1, cv2.LINE_AA, shift)

r = 3.0

cv2.circle(display, tuple(map(int, (p0*pow2+0.5))), int(r*pow2+0.5),
           (0, 255, 0), 1, cv2.LINE_AA, shift)

cv2.circle(display, tuple(map(int, (p1*pow2+0.5))), int(r*pow2+0.5),
           (255, 0, 255), 1, cv2.LINE_AA, shift)

cv2.circle(display, tuple(map(int, (Hp0*pow2+0.5))), int(r*pow2+0.5),
           (0, 0, 255), 1, cv2.LINE_AA, shift)

cv2.imshow('win', display)
while cv2.waitKey(5) < 0: pass


