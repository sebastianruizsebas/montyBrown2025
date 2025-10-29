import os
from tbp.monty.frameworks.utils.logging_utils import load_stats

exp_path = os.path.expanduser("~/data/sruiz10/tbp/results/monty/projects/surf_agent_1lm_6obj/eval_9")

_, _, detailed_stats, _ = load_stats(
    exp_path=exp_path,
    train_stats_path=None,
    model_path_file=None,
    load_train=False,
    load_eval=False,
    load_detailed=True,
    load_models=False,
)

# Print the structure
print("Top-level keys:", list(detailed_stats.keys())[:5])  # First 5 episodes

if detailed_stats:
    first_episode = detailed_stats[list(detailed_stats.keys())[0]]
    print("\nFirst episode keys:", list(first_episode.keys()))
    
    if "LM_0" in first_episode:
        print("LM_0 keys:", list(first_episode["LM_0"].keys()))
        
        # Print all keys recursively
        for key, value in first_episode["LM_0"].items():
            if isinstance(value, dict):
                print(f"  {key}: {list(value.keys())[:5]}")
            elif isinstance(value, list) and value:
                print(f"  {key}: list with {len(value)} items")
                if isinstance(value[0], dict):
                    print(f"    First item keys: {list(value[0].keys())}")
            else:
                print(f"  {key}: {type(value)}")