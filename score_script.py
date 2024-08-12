import json
import os

# Build the path to the JSON file
training_sols = "arc-agi_training_solutions.json"
evaluations_sols = "arc-agi_evaluation_solutions.json"
file_name = "submission.json"
directory_path = "data"


def find_file(directory_path, file_name):
    # Create the full path to the file within the current working directory
    full_path = os.path.join(os.getcwd(), directory_path, file_name)
    return os.path.isfile(full_path)

# test submission.json is present
assert find_file(directory_path=directory_path, file_name=file_name), "The submission.json file has not been found"

# open solution files and store in dict
solution_dict = {}
with open(os.path.join(os.getcwd(), directory_path, training_sols),"r") as f:
    solution_dict = json.load(f)
with open(os.path.join(os.getcwd(), directory_path, evaluations_sols),"r") as f:
    solution_dict = solution_dict | json.load(f)

assert len(solution_dict) == 800, "something of when loading solution files"

# score function
def score():
    """scoring function of the submitted solution"""
    data = {}
    with open(os.path.join(os.getcwd(), directory_path, file_name), "r") as file:
        data = json.load(file)  # proposed solutions
    attempted, score = 0, 0
    for task_name, attempts in data.items():
        attempted += 1
        local_score = 0
        for j, pair_of_attempts in enumerate(attempts):
            try:
                grid_expected = solution_dict[task_name][j]
            except KeyError:
                print(f"the filename {task_name} does not exist among the training and evaluation tasks: this is a bug")
            if (data[task_name][j]["attempt_1"] == grid_expected) or (
                data[task_name][j]["attempt_2"] == grid_expected
            ):
                local_score += 1
        score += local_score / len(solution_dict[task_name])
    return attempted, score, score / attempted

print("This is the total score:", score())
