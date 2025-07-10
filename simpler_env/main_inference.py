import os

import numpy as np
import tensorflow as tf

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator

if __name__ == "__main__":
    # # Set random seeds for reproducibility
    # np.random.seed(0)
    # tf.random.set_seed(0)
    # os.environ['PYTHONHASHSEED'] = '0'
    # import random
    # random.seed(0)
    
    args = get_args()

    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )
    print(f"**** {args.policy_model} ****")
    # policy model creation; update this if you are using a new policy model

    if args.policy_model == "sofar_widowx":
        from simpler_env.evaluation.maniskill2_evaluator_sofar_widowx import maniskill2_evaluator_sofar_widowx
        success_arr = maniskill2_evaluator_sofar_widowx(args.policy_model, args)
    elif args.policy_model == "sofar":
        from simpler_env.evaluation.maniskill2_evaluator_sofar import maniskill2_evaluator_sofar
        success_arr = maniskill2_evaluator_sofar(args.policy_model, args)
    elif args.policy_model == "humanpoint_widowx":
        from simpler_env.evaluation.maniskill2_evaluator_humanpoint_widowx import maniskill2_evaluator_humanpoint_widowx
        success_arr = maniskill2_evaluator_humanpoint_widowx(args.policy_model, args)
    elif args.policy_model == "humanpoint":
        from simpler_env.evaluation.maniskill2_evaluator_humanpoint import maniskill2_evaluator_humanpoint
        success_arr = maniskill2_evaluator_humanpoint(args.policy_model, args)
    elif args.policy_model == "fsd_widowx":
        from simpler_env.evaluation.maniskill2_evaluator_fsd_widowx import maniskill2_evaluator_fsd_widowx
        success_arr = maniskill2_evaluator_fsd_widowx(args.policy_model, args)
    elif args.policy_model == "fsd":
        from simpler_env.evaluation.maniskill2_evaluator_fsd import maniskill2_evaluator_fsd
        success_arr = maniskill2_evaluator_fsd(args.policy_model, args)
    elif args.policy_model == "robopoint_widowx":
        from simpler_env.evaluation.maniskill2_evaluator_robopoint_widowx import maniskill2_evaluator_robopoint_widowx
        success_arr = maniskill2_evaluator_robopoint_widowx(args.policy_model, args)
    elif args.policy_model == "robopoint":
        from simpler_env.evaluation.maniskill2_evaluator_robopoint import maniskill2_evaluator_robopoint
        success_arr = maniskill2_evaluator_robopoint(args.policy_model, args)
    else:
        success_arr = maniskill2_evaluator(args.policy_model, args)

    # run real-to-sim evaluation
    print(" " * 10, "Average success", np.mean(success_arr))
