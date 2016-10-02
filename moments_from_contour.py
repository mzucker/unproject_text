'''The function below is ported from the OpenCV project's
contourMoments function in opencv/modules/imgproc/src/moments.cpp,
licensed as follows:

----------------------------------------------------------------------

By downloading, copying, installing or using the software you agree to
this license.  If you do not agree to this license, do not download,
install, copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2016, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

  * Neither the names of the copyright holders nor the names of the
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

This software is provided by the copyright holders and contributors
"as is" and any express or implied warranties, including, but not
limited to, the implied warranties of merchantability and fitness for
a particular purpose are disclaimed.  In no event shall copyright
holders or contributors be liable for any direct, indirect,
incidental, special, exemplary, or consequential damages (including,
but not limited to, procurement of substitute goods or services; loss
of use, data, or profits; or business interruption) however caused and
on any theory of liability, whether in contract, strict liability, or
tort (including negligence or otherwise) arising in any way out of the
use of this software, even if advised of the possibility of such
damage.

'''


def moments_from_contour(xypoints):

    '''Create shape moments from points sampled from the outline of an
ellipse (note this is numerically inaccurate even for arrays of 1000s
of points). Included in this project primarily for testing purposes.

    '''

    assert len(xypoints.shape) == 3
    assert xypoints.shape[1:] == (1, 2)

    xypoints = xypoints.reshape((-1, 2))

    a00 = 0
    a10 = 0
    a01 = 0
    a20 = 0
    a11 = 0
    a02 = 0

    xi_1, yi_1 = xypoints[-1]

    for xy in xypoints:

        xi, yi = xy
        xi2 = xi * xi
        yi2 = yi * yi
        dxy = xi_1 * yi - xi * yi_1
        xii_1 = xi_1 + xi
        yii_1 = yi_1 + yi

        a00 += dxy
        a10 += dxy * xii_1
        a01 += dxy * yii_1
        a20 += dxy * (xi_1 * xii_1 + xi2)
        a11 += dxy * (xi_1 * (yii_1 + yi_1) + xi * (yii_1 + yi))
        a02 += dxy * (yi_1 * yii_1 + yi2)

        xi_1 = xi
        yi_1 = yi

    if a00 > 0:
        db1_2 = 0.5
        db1_6 = 0.16666666666666666666666666666667
        db1_12 = 0.083333333333333333333333333333333
        db1_24 = 0.041666666666666666666666666666667
    else:
        db1_2 = -0.5
        db1_6 = -0.16666666666666666666666666666667
        db1_12 = -0.083333333333333333333333333333333
        db1_24 = -0.041666666666666666666666666666667

    m00 = a00 * db1_2
    m10 = a10 * db1_6
    m01 = a01 * db1_6
    m20 = a20 * db1_12
    m11 = a11 * db1_24
    m02 = a02 * db1_12

    inv_m00 = 1. / m00
    cx = m10 * inv_m00
    cy = m01 * inv_m00

    mu20 = m20 - m10 * cx
    mu11 = m11 - m10 * cy
    mu02 = m02 - m01 * cy

    return m00, m10, m01, mu20, mu11, mu02
