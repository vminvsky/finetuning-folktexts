import os 
import json 
from folktexts.acs import ACSDataset
from folktexts.acs import ACSTaskMetadata
from folktexts.acs.acs_thresholds import acs_income_threshold, acs_public_coverage_threshold, acs_mobility_threshold, acs_employment_threshold, acs_travel_time_threshold, acs_poverty_ratio_threshold
from folktexts.prompting import encode_row_prompt


def return_training_prompts(acs_task_name="ACSIncome"):
    dataset = ACSDataset.make_from_task(acs_task_name, cache_dir="/scratch/gpfs/vv7118/projects/finetuning-folktexts/data/")
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = dataset.get_train(), dataset.get_val(), dataset.get_test()

    # limit train to 10000 examples
    train_X = train_X.iloc[:10000]
    train_y = train_y.iloc[:10000]

    # limit val to 1000 examples
    val_X = val_X.iloc[:1000]
    val_y = val_y.iloc[:1000]

    # limit test to 1000 examples
    test_X = test_X.iloc[:1000]
    test_y = test_y.iloc[:1000]

    if acs_task_name == "ACSIncome":
        acs_income_task = ACSTaskMetadata.make_folktables_task(
            name=acs_task_name,
            target_threshold=acs_income_threshold,
            description="predict whether an individual's income is above $50,000",
        )
        
    elif acs_task_name == "ACSPublicCoverage":
        acs_income_task = ACSTaskMetadata.make_folktables_task(
            name=acs_task_name,
            target_threshold=acs_public_coverage_threshold,
            description="predict whether an individual is covered by public health insurance",
        )

    elif acs_task_name == "ACSMobility":
        acs_income_task = ACSTaskMetadata.make_folktables_task(
            name=acs_task_name,
            target_threshold=acs_mobility_threshold,
            description="predict whether an individual had the same residential address one year ago",
        )

    elif acs_task_name == "ACSEmployment":
        acs_income_task = ACSTaskMetadata.make_folktables_task(
            name=acs_task_name,
            target_threshold=acs_employment_threshold,
            description="predict whether an individual is employed",
        )

    elif acs_task_name == "ACSTravelTime":
        acs_income_task = ACSTaskMetadata.make_folktables_task(
            name=acs_task_name,
            target_threshold=acs_travel_time_threshold,
            description="predict whether an individual has a commute to work that is longer than 20 minutes",
        )

    elif acs_task_name == "ACSIncomePovertyRatio":
        acs_income_task = ACSTaskMetadata.make_folktables_task(
            name=acs_task_name,
            target_threshold=acs_poverty_ratio_threshold,
            description="predict whether an individual's income-to-poverty ratio is below 2.5",
        )

    else:
        raise ValueError(f"ACS task name {acs_task_name} not recognized")
    train_prompts = []
    train_labels = []
    for (ir, row), (label) in zip(train_X.iterrows(), train_y):
        prompt = encode_row_prompt(row, acs_income_task)
        train_prompts.append(prompt)
        train_labels.append(int(label))

    val_prompts = []
    val_labels = []
    for (ir, row), (label) in zip(val_X.iterrows(), val_y):
        prompt = encode_row_prompt(row, acs_income_task)
        val_prompts.append(prompt)
        val_labels.append(int(label))

    test_prompts = []
    test_labels = []
    for (ir, row), (label) in zip(test_X.iterrows(), test_y):
        prompt = encode_row_prompt(row, acs_income_task)
        test_prompts.append(prompt)
        test_labels.append(int(label))

    return (train_prompts, train_labels), (val_prompts, val_labels), (test_prompts, test_labels)

def save_as_hf_dataset(data_splits, output_path):
    """
    Save prompts and labels in Hugging Face datasets JSON format.
    Each split is saved as a list of dictionaries with 'text' and 'label' fields.
    """
    (train_prompts, train_labels), (val_prompts, val_labels), (test_prompts, test_labels) = data_splits
    
    dataset = {
        "train": [{"prompt": prompt, "label": label, "text": prompt + " " + str(label)} 
                 for prompt, label in zip(train_prompts, train_labels)],
        "validation": [{"prompt": prompt, "label": label, "text": prompt + " " + str(label)} 
                      for prompt, label in zip(val_prompts, val_labels)],
        "test": [{"prompt": prompt, "label": label, "text": prompt + " " + str(label)} 
                for prompt, label in zip(test_prompts, test_labels)]
    }
    
    for key, value in dataset.items():
        os.makedirs(output_path, exist_ok=True)
        with open(f"{output_path}/{key}.json", 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)

def save_as_instruct_dataset(data_splits, output_path):
    (train_prompts, train_labels), (val_prompts, val_labels), (test_prompts, test_labels) = data_splits

    dataset = {
        "train": [
            {
                'conversations': 
                 [{"from": "human", "value": prompt}, 
                  {"from": "gpt", "value": str(label)}]
                } 
                  for prompt, label in zip(train_prompts, train_labels)
            ],
        "validation": [
            {
                'conversations': 
                 [{"from": "human", "value": prompt}, 
                  {"from": "gpt", "value": str(label)}]
                }
                for prompt, label in zip(val_prompts, val_labels)
            ],
        "test": [
            {
                'conversations': 
                 [
                    {"from": "human", "value": prompt}, 
                    {"from": "gpt", "value": str(label)}
                  ]
                }
                for prompt, label in zip(test_prompts, test_labels)
            ]
        }

    for key, value in dataset.items():
        os.makedirs(output_path, exist_ok=True)
        with open(f"{output_path}/{key}_chat.json", 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)




def main(acs_task_name="ACSIncome", output_path="train_data/"):
    data_splits = return_training_prompts(acs_task_name)
    save_as_hf_dataset(data_splits, output_path)
    save_as_instruct_dataset(data_splits, output_path)
    return data_splits

if __name__ == "__main__":
    prompts = main("ACSIncome")
