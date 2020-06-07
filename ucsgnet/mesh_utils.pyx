import numpy as np
import cython
import math
cimport numpy as np

DFLOAT = np.float32
ctypedef  np.float32_t DFLOAT32_t

DINT32 = np.int32
ctypedef  np.int32_t DINT32_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sample_points_polygon(
    np.ndarray[DFLOAT32_t, ndim=2] vertices,
    np.ndarray[DINT32_t, ndim=2] polygons,
    np.ndarray[DFLOAT32_t, ndim=3] voxel_model,
    int num_of_points,
    int size = 64,
):
    # convert polygons to triangles
    cdef np.ndarray[DINT32_t, ndim=2] triangles = np.empty((len(polygons), 3), dtype=DINT32)

    cdef Py_ssize_t ii, jj, i, j
    for ii in range(len(polygons)):
        for jj in range(len(polygons[ii]) - 2):
            triangles[ii][0] = polygons[ii][0]
            triangles[ii][1] = polygons[ii][jj + 1]
            triangles[ii][2] = polygons[ii][jj + 2]

    cdef float small_step = 1.0 / size
    cdef float epsilon = 1e-6
    cdef np.ndarray[DFLOAT32_t, ndim=1] triangle_area_list = np.zeros([len(triangles)], DFLOAT)
    cdef np.ndarray[DFLOAT32_t, ndim=2] triangle_normal_list = np.zeros([len(triangles), 3], DFLOAT)

    cdef float a, b, c, x, y, z, ti, tj, tk, area2
    for i in range(len(triangles)):
        # area = |u x v|/2 = |u||v|sin(uv)/2
        a, b, c = vertices[triangles[i, 1]] - vertices[triangles[i, 0]]
        x, y, z = vertices[triangles[i, 2]] - vertices[triangles[i, 0]]
        ti = b * z - c * y
        tj = c * x - a * z
        tk = a * y - b * x
        area2 = math.sqrt(ti * ti + tj * tj + tk * tk)
        if area2 < epsilon:
            triangle_area_list[i] = 0
            triangle_normal_list[i, 0] = 0
            triangle_normal_list[i, 1] = 0
            triangle_normal_list[i, 2] = 0
        else:
            triangle_area_list[i] = area2
            triangle_normal_list[i, 0] = ti / area2
            triangle_normal_list[i, 1] = tj / area2
            triangle_normal_list[i, 2] = tk / area2

    cdef float triangle_area_sum = np.sum(triangle_area_list)

    cdef np.ndarray[DFLOAT32_t, ndim=1] sample_prob_list = (num_of_points / triangle_area_sum) * triangle_area_list
    cdef np.ndarray[DINT32_t, ndim=1] triangle_index_list = np.arange(len(triangles)).astype(DINT32)
    cdef np.ndarray[DFLOAT32_t, ndim=2] point_normal_list = np.zeros([num_of_points, 6], DFLOAT)

    cdef int count = 0
    cdef int watchdog = 0

    cdef float u_x, u_y, prob, prob_i, prob_f
    cdef int px1, py1, pz1, ppx, ppy, ppz, dxb

    cdef np.ndarray[DFLOAT32_t, ndim=1] base, ppp, pppn1, u, v
    cdef np.ndarray[DFLOAT32_t, ndim=1] normal_direction
    while count < num_of_points:
        np.random.shuffle(triangle_index_list)
        watchdog += 1
        if watchdog > 100:
            print(
                f"Infinite loop, gather: {count} instead of "
                f"{num_of_points}."
            )
            return point_normal_list
        for i in range(len(triangle_index_list)):
            if count >= num_of_points:
                break
            dxb = triangle_index_list[i]
            prob = sample_prob_list[dxb]
            prob_i = float(int(prob))
            prob_f = prob - prob_i
            if np.random.random() < prob_f:
                prob_i += 1
            normal_direction = triangle_normal_list[dxb]
            u = vertices[triangles[dxb, 1]] - vertices[triangles[dxb, 0]]
            v = vertices[triangles[dxb, 2]] - vertices[triangles[dxb, 0]]
            base = vertices[triangles[dxb, 0]]
            for j in range(int(prob_i)):
                # sample a point here:
                u_x = np.random.random()
                v_y = np.random.random()
                if u_x + v_y >= 1:
                    u_x = 1 - u_x
                    v_y = 1 - v_y
                ppp = u * u_x + v * v_y + base

                # verify normal
                pppn1 = (ppp + normal_direction * small_step + 0.5) * size
                px1 = int(pppn1[0])
                py1 = int(pppn1[1])
                pz1 = int(pppn1[2])

                ppx = int((ppp[0] + 0.5) * size)
                ppy = int((ppp[1] + 0.5) * size)
                ppz = int((ppp[2] + 0.5) * size)

                if (
                    ppx < 0
                    or ppx >= size
                    or ppy < 0
                    or ppy >= size
                    or ppz < 0
                    or ppz >= size
                ):
                    continue
                    # valid
                point_normal_list[count, :3] = ppp
                point_normal_list[count, 3:] = normal_direction
                count += 1
                if count >= num_of_points:
                    break

    return point_normal_list
