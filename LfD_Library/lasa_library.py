import numpy as np
from MP_Library import MP_Library
import h5py
from utils import *
from sim_metrics import *

shape_names = ['Angle', 'BendedLine', 'CShape', 'DoubleBendedLine', 'GShape', 'JShape', 'JShape_2', 'Khamesh', 'LShape', 'Leaf_1', 'Leaf_2', 'Line', 'NShape', 'PShape', 'RShape', 'Saeghe', 'Sharpc', 'Sine', 'Snake', 'Spoon', 'Sshape', 'Trapezoid', 'WShape', 'Worm', 'Zshape', 'heee']

num_demos = 7

def main():
    library = MP_Library(metric=COS_metric, threshold=128.0, debug=True)
    for i in range(num_demos):
        for name in shape_names:
            [x, y] = get_lasa_trajN(name, n=i+1)
            traj = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1))))
            library.add_primitive(traj, name=name + str(i+1))
        library.display()
    library.plot()
    library.plot_separate()
    library.save_h5('lasa_library.h5')
    

if __name__ == '__main__':
    main()