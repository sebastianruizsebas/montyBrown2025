# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

'''
TRAINING CONFIGS 7-28 10:50AM 
Notes: did not produce train_stats.csv.
'''
# import os
# from dataclasses import asdict

# from benchmarks.configs.names import MyExperiments
# from tbp.monty.frameworks.config_utils.config_args import (
#     TwoLMStackedMontyConfig,
#     MontyArgs,
#     MotorSystemConfigNaiveScanSpiral,
#     MotorSystemConfigInformedNoTrans,
#     PretrainLoggingConfig,
#     get_cube_face_and_corner_views_rotations,
# )
# from tbp.monty.frameworks.config_utils.make_dataset_configs import (
#     EnvironmentDataloaderPerObjectArgs,
#     ExperimentArgs,
#     PredefinedObjectInitializer,
#     get_env_dataloader_per_object_by_idx,
# )
# from tbp.monty.frameworks.config_utils.policy_setup_utils import (
#     make_naive_scan_policy_config,
# )
# from tbp.monty.frameworks.environments import embodied_data as ED
# from tbp.monty.frameworks.experiments import (
#     MontySupervisedObjectPretrainingExperiment,
# )
# from tbp.monty.simulators.habitat.configs import (
#     MultiLMMountHabitatDatasetArgs,
# )
# from tbp.monty.frameworks.loggers.wandb_handlers import (
#     BasicWandbTableStatsHandler,
# )
# from tbp.monty.frameworks.loggers.monty_handlers import BasicCSVStatsHandler

# # Specify directory where an output directory will be created.
# project_dir = os.path.expanduser("~/data/sruiz10/tbp/results/monty/projects")

# # Specify a name for the model.
# model_name = "dist_agent_2lm_stack_7-28"

# # Specify the objects to train on and 14 unique object poses.
# object_names = ["mug", "bowl", "c_toy_airplane"]
# train_rotations = get_cube_face_and_corner_views_rotations()

# # The config dictionary for the pretraining experiment.
# dist_agent_2lm_stack_train = dict(
#     # Specify monty experiment class and its args.
#     # The MontySupervisedObjectPretrainingExperiment class will provide the model
#     # with object and pose labels for supervised pretraining.
#     experiment_class=MontySupervisedObjectPretrainingExperiment,
#     experiment_args=ExperimentArgs(
#         do_eval=False,
#         n_train_epochs=len(train_rotations),
#     ),
#     # Specify logging config.
#     logging_config=PretrainLoggingConfig(
#         output_dir=project_dir,
#         run_name=model_name,
#         monty_handlers=[BasicCSVStatsHandler],
#         wandb_handlers=[BasicWandbTableStatsHandler],
#     ),
#     # Specify the Monty model. The FiveLLLMontyConfig contains all of the sensor module
#     # configs, learning module configs, and connectivity matrices we need.
#     monty_config=TwoLMStackedMontyConfig(
#         monty_args=MontyArgs(num_exploratory_steps=500),
#         motor_system_config=MotorSystemConfigInformedNoTrans(),
#     ),
#     # Set up the environment and agent.
#     dataset_class=ED.EnvironmentDataset,
#     dataset_args=MultiLMMountHabitatDatasetArgs(),
#     # Set up the training dataloader.
#     train_dataloader_class=ED.InformedEnvironmentDataLoader,
#     train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
#         object_names=object_names,
#         object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations),
#     ),
#     # Set up the evaluation dataloader. Unused, but required.
#     eval_dataloader_class=ED.InformedEnvironmentDataLoader,  # just placeholder
#     eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
#         object_names=object_names,
#         object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations),
#     ),
# )

# experiments = MyExperiments(
#     dist_agent_2lm_stack_train=dist_agent_2lm_stack_train,
# )
# CONFIGS = asdict(experiments)

'''

TRAINING CONFIGS 7-28 11AM 
Notes: did not produce train_stats.csv.
Modifications from 11am configs:
- Changed model_name to "dist_agent_2lm_stack_7-28-1120am"
- Removed commas from the end of lines in the config dictionary
- Added `do_train=True` to `experiment_args`
Modifications from 1120am configs:
- Changed `num_exploratory_steps` to 1000
- Thought about changing to NaiveScanSpiral, but kept InformedNoTrans
- Changed evaluation dataloader to use `get_env_dataloader_per_object_by_idx(start=0, stop=1)`
Modifications from 1140am to 1240pm configs:
- Changed sensor module configs in config_args to use specific features from Santi's code
- removed monty_args from TwoLMStackedMontyConfig, according to Santi's code
- changed sensor_module_class to FeatureChangeSM in config_args
Modifications from 1240pm to 2:00pm configs:
- Changed connection matrix to disconnect second sensor module from top learning module
- Added `save_raw_obs=True` to the second sensor module config
'''
# import os
# from dataclasses import asdict

# from benchmarks.configs.names import MyExperiments
# from tbp.monty.frameworks.config_utils.config_args import (
#     TwoLMStackedMontyConfig,
#     MontyArgs,
#     MotorSystemConfigInformedNoTrans,
#     PretrainLoggingConfig,
#     get_cube_face_and_corner_views_rotations,
# )
# from tbp.monty.frameworks.config_utils.make_dataset_configs import (
#     EnvironmentDataloaderPerObjectArgs,
#     ExperimentArgs,
#     PredefinedObjectInitializer,
#     get_env_dataloader_per_object_by_idx,
# )
# from tbp.monty.frameworks.config_utils.policy_setup_utils import (
#     make_naive_scan_policy_config,
# )
# from tbp.monty.frameworks.environments import embodied_data as ED
# from tbp.monty.frameworks.experiments import (
#     MontySupervisedObjectPretrainingExperiment,
# )
# from tbp.monty.simulators.habitat.configs import (
#     TwoLMStackedDistantMountHabitatDatasetArgs,
# )
# from tbp.monty.frameworks.loggers.wandb_handlers import (
#     BasicWandbTableStatsHandler,
# )
# from tbp.monty.frameworks.loggers.monty_handlers import BasicCSVStatsHandler

# # Specify directory where an output directory will be created.
# project_dir = os.path.expanduser("~/data/sruiz10/tbp/results/monty/projects")

# # Specify a name for the model.
# model_name = "dist_agent_2lm_stack_7-29-1143am"

# # Specify the objects to train on and 14 unique object poses.
# object_names = ["mug", "bowl", "c_toy_airplane",]
# train_rotations = get_cube_face_and_corner_views_rotations()

# # The config dictionary for the pretraining experiment.
# dist_agent_2lm_stack_train = dict(
#     # Specify monty experiment class and its args.
#     # The MontySupervisedObjectPretrainingExperiment class will provide the model
#     # with object and pose labels for supervised pretraining.
#     experiment_class=MontySupervisedObjectPretrainingExperiment,
#     experiment_args=ExperimentArgs(
#         do_train=True,
#         do_eval=False,
#         n_train_epochs=len(train_rotations),
#         min_lms_match=1, 
#     ),
#     # Specify logging config.
#     logging_config=PretrainLoggingConfig(
#         output_dir=project_dir,
#         run_name=model_name,
#         monty_handlers=[BasicCSVStatsHandler],
#         wandb_handlers=[BasicWandbTableStatsHandler],
#     ),
#     # Specify the Monty model. The FiveLLLMontyConfig contains all of the sensor module
#     # configs, learning module configs, and connectivity matrices we need.
#     monty_config=TwoLMStackedMontyConfig(
#         monty_args=MontyArgs(num_exploratory_steps=1000),
#         motor_system_config=MotorSystemConfigInformedNoTrans(),
#     ),
#     # Set up the environment and agent.
#     dataset_class=ED.EnvironmentDataset,
#     dataset_args=TwoLMStackedDistantMountHabitatDatasetArgs(),
#     # Set up the training dataloader.
#     train_dataloader_class=ED.InformedEnvironmentDataLoader,
#     train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
#         object_names=object_names,
#         object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations),
#     ),
#     # Set up the evaluation dataloader. Unused, but required.
#     eval_dataloader_class=ED.InformedEnvironmentDataLoader,  # just placeholder
#     eval_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=1),
# )

# experiments = MyExperiments(
#     dist_agent_2lm_stack_train=dist_agent_2lm_stack_train,
# )
# CONFIGS = asdict(experiments)


'''
EVALUATION CONFIGS
11:20am
Notes: Saved under 7-28-1240pm did not produce train_stats.csv. Tried to edit the config by removing a bracket block, but it didn't work.
Modifications from 11am configs:
- Changed min_lms_match to 0
11:22am
Modifications from 11:20am configs:
- added connection between second sensor module and top learning module
'''
# import copy
# import os
# import numpy as np
# from dataclasses import asdict

# from benchmarks.configs.names import MyExperiments
# # Add your experiment configurations here
# # e.g.: my_experiment_config = dict(...)


# experiments = MyExperiments(
#     # For each experiment name in MyExperiments, add its corresponding
#     # configuration here.
#     # e.g.: my_experiment=my_experiment_config
# )
# import numpy as np
# from dataclasses import asdict

# from benchmarks.configs.names import MyExperiments

# from tbp.monty.frameworks.config_utils.config_args import (
#     EvalLoggingConfig,
#     TwoLMStackedMontyConfig,
#     MontyArgs,
#     MotorSystemConfigInformedGoalStateDriven,
# )
# from tbp.monty.frameworks.config_utils.make_dataset_configs import (
#     EnvironmentDataloaderPerObjectArgs,
#     EvalExperimentArgs,
#     PredefinedObjectInitializer,
#     get_env_dataloader_per_object_by_idx,
# )

# from tbp.monty.frameworks.environments import embodied_data as ED
# from tbp.monty.frameworks.experiments import (
#     MontyObjectRecognitionExperiment,
# )
# from tbp.monty.frameworks.loggers.monty_handlers import (
#     BasicCSVStatsHandler,
#     ReproduceEpisodeHandler,
# )
# from tbp.monty.frameworks.models.displacement_matching import (
#     DisplacementGraphLM,
# )
# from tbp.monty.frameworks.models.evidence_matching.learning_module import (
#     EvidenceGraphLM,
# )
# from tbp.monty.frameworks.models.evidence_matching.model import (
#     MontyForEvidenceGraphMatching,
# )
# from tbp.monty.frameworks.models.goal_state_generation import (
#     EvidenceGoalStateGenerator,
# )
# from tbp.monty.frameworks.models.sensor_modules import (
#     DetailedLoggingSM,
#     FeatureChangeSM,
# )
# from tbp.monty.simulators.habitat.configs import (
#     MultiLMMountHabitatDatasetArgs,
# )
# from tbp.monty.frameworks.loggers.wandb_handlers import (
#     BasicWandbTableStatsHandler,
# )
# """
# Basic Info
# """

# # Specify directory where an output directory will be created.
# project_dir = os.path.expanduser("~/data/sruiz10/tbp/results/monty/projects")

# # Specify a name for the model.
# model_name = "dist_agent_2lm_stack_7-29-1143am"

# object_names = ["mug", "bowl", "c_toy_airplane",]
# test_rotations = [
#     np.array([0.0, 15.0, 30.0]),
#     np.array([7.0, 77.0, 2.0]),
#     np.array([81.0, 33.0, 90.0]),
# ]

# model_path = os.path.join(
#     project_dir,
#     model_name,
#     "pretrained",
# )

# """
# Learning Module Configs
# """


# # The config dictionary for the pretraining experiment.
# dist__agent_2lm_stack_eval = dict(
#     #  Specify monty experiment class and its args.
#     experiment_class=MontyObjectRecognitionExperiment,
#     experiment_args=EvalExperimentArgs(
#         model_name_or_path=model_path,
#         n_eval_epochs=len(test_rotations),
#         max_total_steps= 5000,
#         min_lms_match=0
        
#     ),
#     logging_config=EvalLoggingConfig(  # Move logging_config inside
#         output_dir=os.path.join(project_dir, model_name),
#         run_name="eval_6",
#         monty_handlers=[BasicCSVStatsHandler],
#         wandb_handlers=[BasicWandbTableStatsHandler],
#     ), # Specify logging config.
#     monty_config=TwoLMStackedMontyConfig(
#         monty_args=MontyArgs(min_eval_steps=100),
#         monty_class=MontyForEvidenceGraphMatching,
#         # Do NOT pass learning_module_configs here
#         motor_system_config=MotorSystemConfigInformedGoalStateDriven(),
#     ),
#     # Set up the environment and agent.
#     dataset_class=ED.EnvironmentDataset,
#     dataset_args=MultiLMMountHabitatDatasetArgs(),
#     # Set up the evaluation dataloader.
#     eval_dataloader_class=ED.InformedEnvironmentDataLoader,
#     eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
#         object_names=object_names,
#         object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
#     ),
#     # Set up the training dataloader. Unused, but must be included.
#     train_dataloader_class=ED.InformedEnvironmentDataLoader,
#     train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
#         object_names=object_names,
#         object_init_sampler=get_env_dataloader_per_object_by_idx(start=0, stop=1),
#     ),
# )

# experiments = MyExperiments(
#     dist_agent_2lm_stack_eval=dist__agent_2lm_stack_eval,
# )
# CONFIGS = asdict(experiments)

# import os
# from dataclasses import asdict

# from benchmarks.configs.names import MyExperiments
# import numpy as np
# from dataclasses import asdict

# from benchmarks.configs.names import MyExperiments

# from tbp.monty.frameworks.config_utils.config_args import (
#     EvalLoggingConfig,
#     TwoLMStackedMontyConfig,
#     MontyArgs,
#     MotorSystemConfigInformedGoalStateDriven,
# )
# from tbp.monty.frameworks.config_utils.make_dataset_configs import (
#     EnvironmentDataloaderPerObjectArgs,
#     EvalExperimentArgs,
#     PredefinedObjectInitializer,
#     get_env_dataloader_per_object_by_idx,
# )

# from tbp.monty.frameworks.environments import embodied_data as ED
# from tbp.monty.frameworks.experiments import (
#     MontyObjectRecognitionExperiment,
# )
# from tbp.monty.frameworks.loggers.monty_handlers import (
#     BasicCSVStatsHandler,
#     ReproduceEpisodeHandler,
# )
# from tbp.monty.frameworks.models.displacement_matching import (
#     DisplacementGraphLM,
# )
# from tbp.monty.frameworks.models.evidence_matching.learning_module import (
#     EvidenceGraphLM,
# )
# from tbp.monty.frameworks.models.evidence_matching.model import (
#     MontyForEvidenceGraphMatching,
# )
# from tbp.monty.frameworks.models.goal_state_generation import (
#     EvidenceGoalStateGenerator,
# )
# from tbp.monty.frameworks.models.sensor_modules import (
#     DetailedLoggingSM,
#     FeatureChangeSM,
# )
# from tbp.monty.simulators.habitat.configs import (
#     MultiLMMountHabitatDatasetArgs,
# )
# from tbp.monty.frameworks.loggers.wandb_handlers import (
#     BasicWandbTableStatsHandler,
# )
# """
# Basic Info
# """

# # Specify directory where an output directory will be created.
# project_dir = os.path.expanduser("~/data/sruiz10/tbp/results/monty/projects")

# # Specify a name for the model.
# model_name = "dist_agent_2lm_stack_7-29-1143am"

# object_names = ["mug", "bowl", "c_toy_airplane",]
# test_rotations = [
#     np.array([0.0, 15.0, 30.0]),
#     np.array([7.0, 77.0, 2.0]),
#     np.array([81.0, 33.0, 90.0]),
# ]

# model_path = os.path.join(
#     project_dir,
#     model_name,
#     "pretrained",
# )

# """
# Learning Module Configs
# """


# # The config dictionary for the pretraining experiment.
# dist__agent_2lm_stack_eval = dict(
#     #  Specify monty experiment class and its args.
#     experiment_class=MontyObjectRecognitionExperiment,
#     experiment_args=EvalExperimentArgs(
#         model_name_or_path=model_path,
#         n_eval_epochs=len(test_rotations),
#         max_total_steps= 5000,
#         min_lms_match=0
        
#     ),
#     logging_config=EvalLoggingConfig(  # Move logging_config inside
#         output_dir=os.path.join(project_dir, model_name),
#         run_name="eval_6",
#         monty_handlers=[BasicCSVStatsHandler],
#         wandb_handlers=[BasicWandbTableStatsHandler],
#     ), # Specify logging config.
#     monty_config=TwoLMStackedMontyConfig(
#         monty_args=MontyArgs(min_eval_steps=100),
#         monty_class=MontyForEvidenceGraphMatching,
#         # Do NOT pass learning_module_configs here
#         motor_system_config=MotorSystemConfigInformedGoalStateDriven(),
#     ),
#     # Set up the environment and agent.
#     dataset_class=ED.EnvironmentDataset,
#     dataset_args=MultiLMMountHabitatDatasetArgs(),
#     # Set up the evaluation dataloader.
#     eval_dataloader_class=ED.InformedEnvironmentDataLoader,
#     eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
#         object_names=object_names,
#         object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
#     ),
#     # Set up the training dataloader. Unused, but must be included.
#     train_dataloader_class=ED.InformedEnvironmentDataLoader,
#     train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
#         object_names=object_names,
#         object_init_sampler=get_env_dataloader_per_object_by_idx(start=0, stop=1),
#     ),
# )

# experiments = MyExperiments(
#     dist_agent_2lm_stack_eval=dist__agent_2lm_stack_eval,
# )
# CONFIGS = asdict(experiments)

'''
Noise experiment to be modeled with Drift Diffusion Model
NOTE: This is the pretraining config from tutorial 2, to be ran first before performing inference with various noise levels.
'''
'''
import os
from dataclasses import asdict

from benchmarks.configs.names import MyExperiments
from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    MotorSystemConfigCurvatureInformedSurface,
    PatchAndViewMontyConfig,
    PretrainLoggingConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.models.graph_matching import GraphLM
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    HabitatSurfacePatchSM,
)
from tbp.monty.simulators.habitat.configs import (
    SurfaceViewFinderMountHabitatDatasetArgs,
)

"""
Basic setup
-----------
"""
# Specify directory where an output directory will be created.
project_dir = os.path.expanduser("~/tbp/results/monty/projects")

# Specify a name for the model.
model_name = "surf_agent_1lm_6obj"

"""
Training
----------------------------------------------------------------------------------------
"""
# Here we specify which objects to learn. 'mug' and 'banana' come from the YCB dataset.
# If you don't have the YCB dataset, replace with names from habitat (e.g.,
# 'capsule3DSolid', 'cubeSolid', etc.).
object_names = ["mug","c_cups", "fork", "spoon", "knife", "banana"]
# Get predefined object rotations that give good views of the object from 14 angles.
train_rotations = get_cube_face_and_corner_views_rotations()

# The config dictionary for the pretraining experiment.
surf_agent_6obj_train_10_12_2025 = dict(
    # Specify monty experiment and its args.
    # The MontySupervisedObjectPretrainingExperiment class will provide the model
    # with object and pose labels for supervised pretraining.
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=len(train_rotations),
        do_eval=False,
    ),
    # Specify logging config.
    logging_config=PretrainLoggingConfig(
        output_dir=project_dir,
        run_name=model_name,
        wandb_handlers=[],
    ),
    # Specify the Monty config.
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=500),
        # sensory module configs: one surface patch for training (sensor_module_0),
        # and one view-finder for initializing each episode and logging
        # (sensor_module_1).
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatSurfacePatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    # a list of features that the SM will extract and send to the LM
                    features=[
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "min_depth",
                        "mean_depth",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        # learning module config: 1 graph learning module.
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=GraphLM,
                learning_module_args=dict(),  # Use default LM args
            )
        ),
        # Motor system config: class specific to surface agent.
        motor_system_config=MotorSystemConfigCurvatureInformedSurface(),
    ),
    # Set up the environment and agent
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations),
    ),
    # For a complete config we need to specify an eval_dataloader but since we only train here, this is unused
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations),
    ),
)
experiments = MyExperiments(
    surf_agent_6obj_train_10_12_2025=surf_agent_6obj_train_10_12_2025,
)
CONFIGS = asdict(experiments)
'''

# Eval config for surf_agent_1lm_6obj pretrained model
# TODO: Rerun with copy and pasting into notebook for documentation
'''
import os
from dataclasses import asdict

import numpy as np

from benchmarks.configs.names import MyExperiments
from tbp.monty.frameworks.config_utils.config_args import (
    DetailedEvidenceLMLoggingConfig,
    MontyArgs,
    MotorSystemConfigCurInformedSurfaceGoalStateDriven,
    PatchAndViewSOTAMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
    MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM
)
from tbp.monty.frameworks.models.goal_state_generation import (
    EvidenceGoalStateGenerator,
)
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    FeatureChangeSM,
)
from tbp.monty.simulators.habitat.configs import (
    SurfaceViewFinderMountHabitatDatasetArgs,
)

"""
Basic setup
-----------
"""
# Specify the directory where an output directory will be created.
project_dir = os.path.expanduser("~/data/sruiz10/tbp/results/monty/projects")

# Specify the model name. This needs to be the same name as used for pretraining.
model_name = "surf_agent_1lm_6obj"

# Where to find the pretrained model.
model_path = os.path.join(project_dir, model_name, "pretrained")

# Where to save eval logs.
output_dir = os.path.join(project_dir, model_name)
run_name = "eval"

# Specify objects to test and the rotations in which they'll be presented.
object_names = ["mug", "c_cups", "fork", "spoon", "knife", "banana"]
test_rotations = [
    np.array([0.0, 15.0, 30.0]),
    np.array([7.0, 77.0, 2.0]),
    np.array([81.0, 33.0, 90.0]),
]

# Let's add some noise to the sensor module outputs to make the task more challenging.
sensor_noise_params = dict(
    features=dict(
        pose_vectors=2,  # rotate by random degrees along xyz
        hsv=np.array([0.1, 0.1, 0.1]),  # add noise to each channel (the values here specify std. deviation of gaussian for each channel individually)
        principal_curvatures_log=0.1,
        pose_fully_defined=0.01,  # flip bool in 1% of cases
    ),
    location=0.002,  # add gaussian noise with 0.002 std (0.2cm)
)

sensor_module_0 = dict(
    sensor_module_class=FeatureChangeSM,
    sensor_module_args=dict(
        sensor_module_id="patch",
        # Features that will be extracted and sent to LM
        # note: don't have to be all the features extracted during pretraining.
        features=[
            "pose_vectors",
            "pose_fully_defined",
            "on_object",
            "object_coverage",
            "min_depth",
            "mean_depth",
            "hsv",
            "principal_curvatures",
            "principal_curvatures_log",
        ],
        save_raw_obs=False,
        # FeatureChangeSM will only send an observation to the LM if features or location
        # changed more than these amounts.
        delta_thresholds={
            "on_object": 0,
            "n_steps": 20,
            "hsv": [0.1, 0.1, 0.1],
            "pose_vectors": [np.pi / 4, np.pi * 2, np.pi * 2],
            "principal_curvatures_log": [2, 2],
            "distance": 0.01,
        },
        surf_agent_sm=True,  # for surface agent
        noise_params=sensor_noise_params,
    ),
)
sensor_module_1 = dict(
    sensor_module_class=DetailedLoggingSM,
    sensor_module_args=dict(
        sensor_module_id="view_finder",
        save_raw_obs=False,
    ),
)
sensor_module_configs = dict(
    sensor_module_0=sensor_module_0,
    sensor_module_1=sensor_module_1,
)

# Tolerances within which features must match stored values in order to add evidence
# to a hypothesis.
tolerances = {
    "patch": {
        "hsv": np.array([0.1, 0.2, 0.2]),
        "principal_curvatures_log": np.ones(2),
    }
}

# Features where weight is not specified default to 1.
feature_weights = {
    "patch": {
        # Weighting saturation and value less since these might change under different
        # lighting conditions.
        "hsv": np.array([1, 0.5, 0.5]),
    }
}

learning_module_0 = dict(
    learning_module_class=EvidenceGraphLM,
    learning_module_args=dict(
        # Search the model in a radius of 1cm from the hypothesized location on the model.
        max_match_distance=0.01,  # =1cm
        tolerances=tolerances,
        feature_weights=feature_weights,
        # Most likely hypothesis needs to have 20% more evidence than the others to 
        # be considered certain enough to trigger a terminal condition (match).
        x_percent_threshold=20,
        # Update all hypotheses with evidence > x_percent_threshold (faster)
        evidence_threshold_config="x_percent_threshold",
        # Config for goal state generator of LM which is used for model-based action
        # suggestions, such as hypothesis-testing actions.
        gsg_class=EvidenceGoalStateGenerator,
        gsg_args=dict(
            # Tolerance(s) when determining goal-state success
            goal_tolerances=dict(
                location=0.015,  # distance in meters
            ),
            # Number of necessary steps for a hypothesis-testing action to be considered
            min_post_goal_success_steps=5,
        ),
        hypotheses_updater_args=dict(
            # Look at features associated with (at most) the 10 closest learned points.
            max_nneighbors=10,
        )
    ),
)
learning_module_configs = dict(learning_module_0=learning_module_0)

# The config dictionary for the evaluation experiment.
surf_agent_6obj_eval = dict(
    # Set up experiment
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,  # load the pre-trained models from this path
        n_eval_epochs=len(test_rotations),
        max_total_steps=5000,
    ),
    logging_config=DetailedEvidenceLMLoggingConfig(
        output_dir=output_dir,
        run_name=run_name,
        wandb_handlers=[],  # remove this line if you, additionally, want to log to WandB.
    ),
    # Set up monty, including LM, SM, and motor system.
    monty_config=PatchAndViewSOTAMontyConfig(
        monty_args=MontyArgs(min_eval_steps=20),
        sensor_module_configs=sensor_module_configs,
        learning_module_configs=learning_module_configs,
        motor_system_config=MotorSystemConfigCurInformedSurfaceGoalStateDriven(),
    ),
    # Set up environment/data
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
    # Doesn't get used, but currently needs to be set anyways.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
)
experiments = MyExperiments(
    surf_agent_6obj_eval=surf_agent_6obj_eval,
)
CONFIGS = asdict(experiments)
'''

# Eval config for surf_agent_1lm_6obj pretrained model
# Noise = [0.x, 0.x, 0.x] for hsv
# run_name = "eval_x" where x is the noise level
import os
from dataclasses import asdict

import numpy as np

from benchmarks.configs.names import MyExperiments
from tbp.monty.frameworks.config_utils.config_args import (
    DetailedEvidenceLMLoggingConfig,
    MontyArgs,
    MotorSystemConfigCurInformedSurfaceGoalStateDriven,
    PatchAndViewSOTAMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
    MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM
)
from tbp.monty.frameworks.models.goal_state_generation import (
    EvidenceGoalStateGenerator,
)
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    FeatureChangeSM,
)
from tbp.monty.simulators.habitat.configs import (
    SurfaceViewFinderMountHabitatDatasetArgs,
)

"""
Basic setup
-----------
"""
# Specify the directory where an output directory will be created.
project_dir = os.path.expanduser("~/data/sruiz10/tbp/results/monty/projects")

# Specify the model name. This needs to be the same name as used for pretraining.
model_name = "surf_agent_1lm_6obj"

# Where to find the pretrained model.
model_path = os.path.join(project_dir, model_name, "pretrained")

# Where to save eval logs.
output_dir = os.path.join(project_dir, model_name, "10_29_2025_noise_eval")
run_name = "eval_1"

# Specify objects to test and the rotations in which they'll be presented.
object_names = ["mug", "c_cups", "fork", "spoon", "knife", "banana"]
test_rotations = [
    np.array([0.0, 15.0, 30.0]),
    np.array([7.0, 77.0, 2.0]),
    np.array([81.0, 33.0, 90.0]),
]

# Let's add some noise to the sensor module outputs to make the task more challenging.
sensor_noise_params = dict(
    features=dict(
        pose_vectors=2,  # rotate by random degrees along xyz
        hsv=np.array([0.1, 0.1, 0.1]),  # add noise to each channel (the values here specify std. deviation of gaussian for each channel individually)
        principal_curvatures_log=0.1,
        pose_fully_defined=0.01,  # flip bool in 1% of cases
    ),
    location=0.002,  # add gaussian noise with 0.002 std (0.2cm)
)

sensor_module_0 = dict(
    sensor_module_class=FeatureChangeSM,
    sensor_module_args=dict(
        sensor_module_id="patch",
        # Features that will be extracted and sent to LM
        # note: don't have to be all the features extracted during pretraining.
        features=[
            "pose_vectors",
            "pose_fully_defined",
            "on_object",
            "object_coverage",
            "min_depth",
            "mean_depth",
            "hsv",
            "principal_curvatures",
            "principal_curvatures_log",
        ],
        save_raw_obs=False,
        # FeatureChangeSM will only send an observation to the LM if features or location
        # changed more than these amounts.
        delta_thresholds={
            "on_object": 0,
            "n_steps": 20,
            "hsv": [0.1, 0.1, 0.1],
            "pose_vectors": [np.pi / 4, np.pi * 2, np.pi * 2],
            "principal_curvatures_log": [2, 2],
            "distance": 0.01,
        },
        surf_agent_sm=True,  # for surface agent
        noise_params=sensor_noise_params,
    ),
)
sensor_module_1 = dict(
    sensor_module_class=DetailedLoggingSM,
    sensor_module_args=dict(
        sensor_module_id="view_finder",
        save_raw_obs=False,
    ),
)
sensor_module_configs = dict(
    sensor_module_0=sensor_module_0,
    sensor_module_1=sensor_module_1,
)

# Tolerances within which features must match stored values in order to add evidence
# to a hypothesis.
tolerances = {
    "patch": {
        "hsv": np.array([0.1, 0.2, 0.2]),
        "principal_curvatures_log": np.ones(2),
    }
}

# Features where weight is not specified default to 1.
feature_weights = {
    "patch": {
        # Weighting saturation and value less since these might change under different
        # lighting conditions.
        "hsv": np.array([1, 0.5, 0.5]),
    }
}

learning_module_0 = dict(
    learning_module_class=EvidenceGraphLM,
    learning_module_args=dict(
        # Search the model in a radius of 1cm from the hypothesized location on the model.
        max_match_distance=0.01,  # =1cm
        tolerances=tolerances,
        feature_weights=feature_weights,
        # Most likely hypothesis needs to have 20% more evidence than the others to 
        # be considered certain enough to trigger a terminal condition (match).
        x_percent_threshold=20,
        # Update all hypotheses with evidence > x_percent_threshold (faster)
        evidence_threshold_config="x_percent_threshold",
        # Config for goal state generator of LM which is used for model-based action
        # suggestions, such as hypothesis-testing actions.
        gsg_class=EvidenceGoalStateGenerator,
        gsg_args=dict(
            # Tolerance(s) when determining goal-state success
            goal_tolerances=dict(
                location=0.015,  # distance in meters
            ),
            # Number of necessary steps for a hypothesis-testing action to be considered
            min_post_goal_success_steps=5,
        ),
        hypotheses_updater_args=dict(
            # Look at features associated with (at most) the 10 closest learned points.
            max_nneighbors=10,
        )
    ),
)
learning_module_configs = dict(learning_module_0=learning_module_0)

# The config dictionary for the evaluation experiment.
surf_agent_6obj_eval_10_29 = dict(
    # Set up experiment
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,  # load the pre-trained models from this path
        n_eval_epochs=len(test_rotations),
        max_total_steps=5000,
    ),
    logging_config=DetailedEvidenceLMLoggingConfig(
        output_dir=output_dir,
        run_name=run_name,
        wandb_handlers=[],  # remove this line if you, additionally, want to log to WandB.
    ),
    # Set up monty, including LM, SM, and motor system.
    monty_config=PatchAndViewSOTAMontyConfig(
        monty_args=MontyArgs(min_eval_steps=20),
        sensor_module_configs=sensor_module_configs,
        learning_module_configs=learning_module_configs,
        motor_system_config=MotorSystemConfigCurInformedSurfaceGoalStateDriven(),
    ),
    # Set up environment/data
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
    # Doesn't get used, but currently needs to be set anyways.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
)
experiments = MyExperiments(
    surf_agent_6obj_eval_10_29=surf_agent_6obj_eval_10_29,
)
CONFIGS = asdict(experiments)