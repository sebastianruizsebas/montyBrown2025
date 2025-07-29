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
import copy
import os

import numpy as np
from dataclasses import asdict

from benchmarks.configs.names import MyExperiments

from tbp.monty.frameworks.config_utils.config_args import (
    EvalLoggingConfig,
    TwoLMStackedMontyConfig,
    MontyArgs,
    MotorSystemConfigInformedGoalStateDriven,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
    get_env_dataloader_per_object_by_idx,
)

from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
    MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.loggers.monty_handlers import (
    BasicCSVStatsHandler,
    ReproduceEpisodeHandler,
)
from tbp.monty.frameworks.models.displacement_matching import (
    DisplacementGraphLM,
)
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.goal_state_generation import (
    EvidenceGoalStateGenerator,
)
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    FeatureChangeSM,
)
from tbp.monty.simulators.habitat.configs import (
    MultiLMMountHabitatDatasetArgs,
)
from tbp.monty.frameworks.loggers.wandb_handlers import (
    BasicWandbTableStatsHandler,
)
"""
Basic Info
"""

# Specify directory where an output directory will be created.
project_dir = os.path.expanduser("~/data/sruiz10/tbp/results/monty/projects")

# Specify a name for the model.
model_name = "dist_agent_2lm_stack_7-29-1143am"

object_names = ["mug", "bowl", "c_toy_airplane",]
test_rotations = [
    np.array([0.0, 15.0, 30.0]),
    np.array([7.0, 77.0, 2.0]),
    np.array([81.0, 33.0, 90.0]),
]

model_path = os.path.join(
    project_dir,
    model_name,
    "pretrained",
)

"""
Learning Module Configs
"""


# The config dictionary for the pretraining experiment.
dist__agent_2lm_stack_eval = dict(
    #  Specify monty experiment class and its args.
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path,
        n_eval_epochs=len(test_rotations),
        max_total_steps= 5000,
        min_lms_match=0
        
    ),
    logging_config=EvalLoggingConfig(  # Move logging_config inside
        output_dir=os.path.join(project_dir, model_name),
        run_name="eval_6",
        monty_handlers=[BasicCSVStatsHandler],
        wandb_handlers=[BasicWandbTableStatsHandler],
    ), # Specify logging config.
    monty_config=TwoLMStackedMontyConfig(
        monty_args=MontyArgs(min_eval_steps=100),
        monty_class=MontyForEvidenceGraphMatching,
        # Do NOT pass learning_module_configs here
        motor_system_config=MotorSystemConfigInformedGoalStateDriven(),
    ),
    # Set up the environment and agent.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=MultiLMMountHabitatDatasetArgs(),
    # Set up the evaluation dataloader.
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations),
    ),
    # Set up the training dataloader. Unused, but must be included.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=get_env_dataloader_per_object_by_idx(start=0, stop=1),
    ),
)

experiments = MyExperiments(
    dist_agent_2lm_stack_eval=dist__agent_2lm_stack_eval,
)
CONFIGS = asdict(experiments)
