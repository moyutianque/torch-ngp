import cc3d
import numpy as np
import matplotlib.pyplot as plt

def display_example(voxels):
    colors = np.array(
        [
            [0,0,0,0], [1,0,0,0.5], [0,1,0,0.5], [0,0,1,0.5]
        ]
    )
    fcolors = colors[voxels]

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxels, facecolors=fcolors, edgecolor='k')

    plt.show()



def example1():
    # example 1
    labels_in = np.zeros((5, 5, 5), dtype=np.int32)
    labels_in[:2,:2,:2] = 1
    labels_in[-1, -1, -1] = 3
    display_example(labels_in)

    # labels_out, N = cc3d.connected_components(labels_in, connectivity=6, delta=0, return_N=True)

    labels_out, N = cc3d.largest_k(
        labels_in, k=2, 
        connectivity=6, delta=0,
        return_N=True,
    )
    print(N)
    labels_out = labels_in * (labels_out == 1)
    display_example(labels_out)
    print()

def example2_0():
    labels_in = np.zeros((5, 5, 5), dtype=np.int32)
    labels_out, N = cc3d.largest_k(
        labels_in, k=1, 
        connectivity=6, delta=0,
        return_N=True,
    )
    print(N)

def example2():
    # example 2
    labels_in = np.zeros((5, 5, 5), dtype=np.int32)
    labels_in[1:4, 2, 1:4] = 1
    labels_in[0,2,0] = 1
    labels_in[0,1,0] = 1
    labels_in[4,4,4] = 1
    display_example(labels_in)

    labels_out, N = cc3d.largest_k(
        labels_in, k=1, 
        connectivity=6, delta=0,
        return_N=True,
    ) # NOTE careful, the k largest are not labeled by its volume

    print(N)
    for i in range(1, N + 1):
        plot_np = labels_in * (labels_out == i)
        display_example(plot_np)
        print()


if __name__=='__main__':
    example1()
    # example2()
    # example2_0()