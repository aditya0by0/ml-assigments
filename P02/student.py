import pandas as pd
import numpy as np
import sys


def entropy(data):
    _, counts = np.unique(data.iloc[:, -1], return_counts=True)
    probabilities = counts/data.shape[0]
    global num_classes # total number of classes in the original dataset
    return -1 * np.sum(probabilities * np.log(probabilities)/np.log(num_classes))


def cal_info_gain(data, feature):
    pre_split_entropy = entropy(data) # parent entropy
    post_split_entropy = 0
    unique_vals = data[feature].unique()
    # weighted average of entropy of splits 
    for val in unique_vals:
        data_in_branch = data[data[feature] == val]
        post_split_entropy += (len(data_in_branch)/len(data)) * entropy(data_in_branch)
    gain = pre_split_entropy - post_split_entropy
    return gain


def id3(data, depth=0):
    # data consist of only one class
    if len(np.unique(data.iloc[:, -1])) == 1:
        return {
            "depth": depth,
            "attribute": "no_leaf",
            "value": "no_val",
            "entropy": entropy(data),
            "class": data.iloc[0, -1],
        }
    
    # No feature in data (only target feature left)
    if data.shape[1] == 1:
        classes, counts = np.unique(data.iloc[:, -1], return_counts=True)
        return {
            "depth": depth,
            "attribute": "no_leaf",
            "value": "no_leaf",
            "entropy": entropy(data),
            "class": classes[np.argmax(counts)],
        }
    
    # select best attribute
    info_gain = [cal_info_gain(data, "att" + str(i)) for i in range(data.shape[1] - 1)]
    best_attribute = "att" + str(np.argmax(info_gain))
    unique_vals = data[best_attribute].unique()
    
    node = {
        "depth": depth, 
        "attribute": "root", 
        "value": "", 
        "entropy": entropy(data),
        "class": "no_leaf",
        "children": []
    }

    # derive sub-branch 
    for value in sorted(unique_vals):
        subset_data = data[data[best_attribute] == value]
        child_node = id3(subset_data, depth + 1)
        child_node["attribute"] = best_attribute
        child_node["entropy"] = entropy(subset_data)
        child_node["value"] = str("=") + str(value)
        node["children"].append(child_node)
    return node


def print_dt(dtree):
    if dtree["class"] != "no_leaf" :
        print(f"{dtree['depth']},{dtree['attribute']}{dtree['value']},{dtree['entropy']},{dtree['class']}")
    else:
        print(f"{dtree['depth']},{dtree['attribute']}{dtree['value']},{dtree['entropy']},{dtree['class']}")
        for child in dtree["children"]:
            print_dt(child)


if __name__ == "__main__":
    data_file = sys.argv[2]
    data = pd.read_csv(data_file, header=None)
    columns_name = {i:"att" + str(i) for i in range(data.shape[1])}
    data = data.rename(columns=columns_name)
    # to take the log to the base the number of classes available in the entropy formula
    num_classes = len(np.unique(data.iloc[:, -1]))
    decision_tree = id3(data)
    print_dt(decision_tree)
