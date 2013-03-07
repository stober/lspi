"""
"""

import sys, pickle
# from pyrouette.utils.plotutils import scatter_with_graph
# from pyrouette.utils.plotutils import scatter3d_with_graph
import pylab as plt

# Running CSDP
# csdp sdp.dat sdp.sol

# TODO: Write out gram matrix as initial solution. SDPA format?
# TODO: resolve input disparities between SPDA and CSDP?


from isomap import cluster_graph
import numpy as np
import hcluster as hc
import numpy.linalg as la
import pylab
import os

def read_sol_file(filename, size=100):

    # Note: this does not yet support the full SPDA specification.
    pfile = open(filename)

    y = None
    X = np.zeros((size,size))
    Z = np.zeros((size,size))
    for line in pfile:
        values = line.rstrip().split(' ')

        if len(values) > 5:
            y = np.array([float(x) for x in values])

        elif values[0] == '1':
            i,j = int(values[2])-1,int(values[3])-1
            Z[i,j] = float(values[4])
            Z[j,i] = Z[i,j]

        else: # values[0] == '2'

            i,j = int(values[2])-1,int(values[3])-1
            X[i,j] = float(values[4])
            X[j,i] = X[i,j]


    return y,Z,X

def read_sol_file_slack(filename, size=100):

    # Note: this does not yet support the full SPDA specification.
    pfile = open(filename)

    y = None
    X = np.zeros((size,size))
    Z = np.zeros((size,size))
    for line in pfile:
        values = line.rstrip().split(' ')

        if len(values) > 5:
            y = np.array([float(x) for x in values])

        elif values[0] == '1' and values[1] == '1':
            i,j = int(values[2])-1,int(values[3])-1
            Z[i,j] = float(values[4])
            Z[j,i] = Z[i,j]

        elif values[0] == '2' and values[1] == '1':

            i,j = int(values[2])-1,int(values[3])-1
            X[i,j] = float(values[4])
            X[j,i] = X[i,j]

        else:
            pass

    return y,Z,X

# Translate the problem into sparse spda format.
def write_spda_file(filename, m, nblocks, size, c, indx):

    # c1 = 0
    # cij = dij

    # F0 = I
    # F1 = 1 - all ones matrix
    # Fij =

    fp = open(filename, "w")
    fp.write(str(m) + "\n")
    fp.write(str(nblocks) + "\n")
    fp.write(str(size) + " " + str(size) + "\n")

    for i in range(len(c)):
        if i > 0:
            fp.write(" ")
        fp.write(str(c[i]))

    fp.write("\n")

    # write the identity matrix
    for i in range(1,size + 1):
        fp.write( "0 1 %d %d 1.0\n" % (i,i) )

    for i in range(1,size + 1):
        for j in range(i,size + 1):
            fp.write( "1 1 %d %d 1.0\n" % (i,j) )

    cnt = 2
    for (i,j) in indx:

        fp.write( "%d 1 %d %d 1.0\n" % (cnt,  i+1, i+1) )
        fp.write( "%d 1 %d %d 1.0\n" % (cnt,  j+1, j+1) )
        fp.write( "%d 1 %d %d -1.0\n" % (cnt,  i+1, j+1) )
        cnt += 1

    fp.close()


def write_spda_file_slack(filename, m, nblocks, size, c, indx, omega):

    # c1 = 0
    # cij = dij

    # F0 = I
    # F1 = 1 - all ones matrix
    # Fij =

    fp = open(filename, "w")
    fp.write(str(m) + "\n")
    fp.write(str(nblocks) + "\n") # should be 2
    fp.write(str(size) + " " + str(size) + "\n")

    for i in range(len(c)):
        if i > 0:
            fp.write(" ")
        fp.write(str(c[i]))

    fp.write("\n")

    # write the identity matrix
    for i in range(1,size + 1):
        fp.write( "0 1 %d %d 1.0\n" % (i,i) )

    for i in range(1,size + 1):
        for j in range(i,size + 1):
            fp.write( "0 2 %d %d %f\n" % (i,j,-omega))
    # for (i,j) in indx:
    #     fp.write( "0 2 %d %d %f\n" %(i+1, j+1, -omega) ) # the slack variables

    for i in range(1,size + 1):
        for j in range(i,size + 1):
            fp.write( "1 1 %d %d 1.0\n" % (i,j) )

    cnt = 2
    for (i,j) in indx:

        fp.write( "%d 1 %d %d 1.0\n" % (cnt,  i+1, i+1) )
        fp.write( "%d 1 %d %d 1.0\n" % (cnt,  j+1, j+1) )
        fp.write( "%d 1 %d %d -1.0\n" % (cnt,  i+1, j+1) )
        fp.write( "%d 2 %d %d -0.5\n" % (cnt, i+1, j+1 ) ) # the slack variables
        cnt += 1

    fp.close()

def test_embedding(orig_pts, new_pts, indx):
    # compare the embedded distances to the new distances for the embedded pts

    print orig_pts.shape, new_pts.shape
    for (i,j) in indx:
        print i,j
        print (orig_pts[i] - orig_pts[j]).shape, (new_pts[:,i] - new_pts[:,j]).shape
        print la.norm(orig_pts[i] - orig_pts[j]), la.norm(new_pts[:,i] - new_pts[:,j])

def MVU_slack(datafile, dim = 3):
    # takes in a pickled matrix of points - outputs a MVU embedding

    fp = open(datafile)
    pts = pickle.load(fp)
    ans = pickle.load(fp) # latent space coordinates
    size = len(pts)

    k = len(ans[0]) # the number of latent dimensions

    # mean center coordinates
    m = np.mean(pts, axis=0)
    pts = pts - m

    # TODO: move graph cluster algorithm to own file - write in C?

    # compute the distance matrix and cluster
    Y = hc.squareform(hc.pdist(pts,'euclidean'))
    res = cluster_graph(Y, fnc = 'k', size = 8)
    x,y = np.nonzero(res & (Y != 0)) # indices of nearest neighbors

    # generate data to write problem in SPDA format
    # TODO: add slack variable block
    indx = []
    for (i,j) in zip(x,y):
        if i <= j:
            indx.append((i,j))

    m = len(indx) + 1
    nblocks = 2
    c = [0.0]
    for (i,j) in indx:
        c.append(Y[i,j]**2)

    write_spda_file_slack("../ds/sdp.dat", m, nblocks, size, c, indx, .01)

    # TODO: add some error checking
    os.system("csdp ../ds/sdp.dat ../ds/sdp.sol")

    y,Z,X = read_sol_file_slack("../ds/sdp.sol", size)

    # spectral decomposition of the dual solution (X)
    u,s,v = la.svd(X)

    results = []
    for i in range(dim):
        results.append(np.sqrt(s[i]) * u[:,i])

    # returns the neighborhood graph for proper plotting
    return results, pts, res

def MVU(ptsfile, ansfile, sdpfile = '../ds/sdp', dim = 3):
    # takes in a pickled matrix of points - outputs a MVU embedding

    pts = np.load(ptsfile)['arr_0']
    ans = np.load(ansfile)['arr_0']
    size = len(pts)

    k = len(ans[0]) # the number of latent dimensions

    # mean center coordinates
    m = np.mean(pts, axis=0)
    pts = pts - m

    # TODO: move graph cluster algorithm to own file - write in C?

    # compute the distance matrix and cluster
    Y = hc.squareform(hc.pdist(pts,'euclidean'))
    res = cluster_graph(Y, fnc = 'k', size = 8)
    x,y = np.nonzero(res & (Y != 0)) # indices of nearest neighbors

    # generate data to write problem in SPDA format
    # TODO: add slack variable block
    indx = []
    for (i,j) in zip(x,y):
        if i <= j:
            indx.append((i,j))

    m = len(indx) + 1
    nblocks = 1
    c = [0.0]
    for (i,j) in indx:
        c.append(Y[i,j]**2)

    write_spda_file(sdpfile + ".dat", m, nblocks, size, c, indx)

    # TODO: add some error checking
    os.system("csdp " + sdpfile + ".dat " + sdpfile + ".sol")

    y,Z,X = read_sol_file(sdpfile + ".sol", size)

    # spectral decomposition of the dual solution (X)
    u,s,v = la.svd(X)

    results = []
    for i in range(dim):
        results.append(np.sqrt(s[i]) * u[:,i])

    # returns the neighborhood graph for proper plotting
    return results, pts, res

if __name__ == '__main__':

    if True:

        # results, pts, res = MVU('../ds/swiss_roll_hd.npz',
        #                         '../ds/swiss_roll_latent.npz',
        #                         sdpfile='../ds/swiss_roll_sdp')

        # np.savez("../ds/swiss_roll_results.npz",results)
        # np.savez("../ds/swiss_roll_pts.npz",pts)
        # np.savez("../ds/swiss_roll_res.npz",res)


        results, pts, res = MVU('../ds/swiss_roll_hd_test.npz',
                                '../ds/swiss_roll_latent_test.npz',
                                sdpfile='../ds/swiss_roll_sdp_test')

        np.savez("../ds/swiss_roll_results_test.npz",results)
        np.savez("../ds/swiss_roll_pts_test.npz",pts)
        np.savez("../ds/swiss_roll_res_test.npz",res)


        # ax1 = scatter3d_with_graph(pts[:,0],pts[:,1],pts[:,2],res)
        # ax2 = scatter3d_with_graph(results[0],results[1],results[2],res)

        # ax2.set_xlim3d(-5,5)
        # ax2.set_ylim3d(-1,1)
        # ax2.set_zlim3d(-1,1)
        # plt.show()

    if False:

        results, pts, res = MVU_slack('../ds/half_ring.pck',dim=2)
        ax1 = scatter_with_graph(pts[:,0],pts[:,1],res)
        ax2 = scatter_with_graph(results[0],results[1],res)
        plt.show()

