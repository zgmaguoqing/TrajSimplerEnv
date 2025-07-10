import pickle as pkl

import sys; sys.path.append("plan")

(pc, robot_pc, obj_thresh) = pkl.load(open("1.pkl", "rb"))
from plan.src.utils.utils import cdist_test


print(cdist_test(pc, robot_pc, obj_thresh))