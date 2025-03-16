import numpy as np
import time
import copy


def startAlg():
    print("Welcome to Amy's Feature Selection Algorithm.")
    
    # get input file name
    filename = input("Please type in the name of the file to test: ")
    # load data into program, source: https://www.geeksforgeeks.org/import-text-files-into-numpy-arrays/
    data = np.loadtxt(filename)
    
    # get number of features = number of columns - 1
    num_features = data.shape[1] - 1
    # number of rows
    num_instances = len(data)
    
    print(f"This dataset has {num_features} features (not including the class attribute), with {num_instances} instances.")
    
    # selection of which algorithm
    print("Type the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    
    choice = input("Selected algorithm: ")

    # nearest neighbor on all features to get initial accuracy, accounting for the index of the column (start at feature 1 not 0)
    initial_accuracy = leave_one_out_cross_validation(data, list(range(1, num_features + 1)), None)
    print(f"Running nearest neighbor with all {num_features} features, using 'leave-one-out' evaluation, I get an accuracy of {initial_accuracy:.1f}%")
    
    # calculate default rate
    labels = data[:, 0]
    # count number of each unique label, source: https://stackoverflow.com/questions/10741346/frequency-counts-for-unique-values-in-a-numpy-array
    unique_labels, counts = np.unique(labels, return_counts=True)
    # get the most common label
    most_common_count = counts.max()
    default_rate = most_common_count / len(labels)

    print(f"Default rate: {default_rate:.1%}")

    # start run time
    start_time = time.time()

    # run the selected algorithm
    if choice == "1":
        best_features, best_accuracy  = forward_selection(data)
    elif choice == "2":
        best_features, best_accuracy = backward_elimination(data)
    else:
        print("Invalid input. Please restart and select 1 or 2.")
        return
    
    # end run time
    end_time = time.time()
    # total run time
    runtime = end_time - start_time

    # runtime in minutes for the small dataset
    minutes = runtime / 60
    # runtime in hours for the large dataset
    hours = runtime / 3600

    # print best features and accuracy
    print(f"Finished search!! The best feature subset is {best_features}, which has an accuracy of {best_accuracy:.1f}%")
    # print runtime
    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Runtime: {minutes:.2f} minutes")
    print(f"Runtime: {hours:.2f} hours")
    
# leave one out cross validation, based on matlab code provided by Professor Keogh in the Project 2 Briefing Slides
# function accuracy = leave_one_out_cross_validation(data, current_set, feature_to_add)
def leave_one_out_cross_validation(data, current_set, feature_to_add):
    num_correctly_classified = 0
    num_instances = data.shape[0]
    
    feature_set_to_use = current_set if feature_to_add is None else current_set + [feature_to_add]

    for i in range(num_instances): 
        object_to_classify = data[i, feature_set_to_use]  # get features, ignore class label
        label_object_to_classify = data[i, 0]  # get class label
        
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_label = None
        
        for k in range(num_instances): 
            if k != i: 
                # source: https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/
                distance = np.linalg.norm(object_to_classify - data[k, feature_set_to_use])  # get euclidean distance
                
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_label = data[k, 0]

        if label_object_to_classify == nearest_neighbor_label:
            num_correctly_classified += 1

    accuracy = (num_correctly_classified / num_instances) * 100

    return accuracy

# forward selection, based on matlab code provided by Professor Keogh in the Project 2 Briefing Slides
# function feature_search_demo(data)
def forward_selection(data):
    num_features = data.shape[1] - 1
    
    # current_set_of_features = [];
    current_set_of_features = []
    best_accuracy = 0 
    best_feature_set = []

    print("\nBeginning Forward Selection\n")

    for i in range(num_features):
        print(f"On the {i+1}th level of the search tree")
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0

        for k in range(1, num_features + 1):
            if k not in current_set_of_features:  # only consider features not already added
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k)

                features_being_tested = current_set_of_features + [k]
                print(f"Using feature(s) {set(features_being_tested)} accuracy is {accuracy:.1f}%")

                if accuracy > best_so_far_accuracy:      
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k


        if feature_to_add_at_this_level is not None:  # add the best feature found
            current_set_of_features.append(feature_to_add_at_this_level)

            print(f"\nFeature set {set(current_set_of_features)} was best, accuracy is {best_so_far_accuracy:.1f}%\n")

            if best_so_far_accuracy > best_accuracy:  # update best accuracy so far
                best_accuracy = best_so_far_accuracy
                best_feature_set = current_set_of_features.copy()

    return best_feature_set, best_accuracy 

# start with full feature set instead of empty set
def backward_elimination(data):
    num_features = data.shape[1] - 1

    # start with all features
    current_set_of_features = list(range(1, num_features + 1))

    # initial best
    best_accuracy = leave_one_out_cross_validation(data, current_set_of_features, None)

    # initial best is all features
    best_feature_set = current_set_of_features.copy()

    print("\nBeginning Backward Elimination\n")

    for i in range(num_features - 1):
        print(f"Backward elimination of features from {current_set_of_features}")
        feature_to_remove_at_this_level = None
        best_so_far_accuracy = best_accuracy

        for k in range(len(current_set_of_features)):   
            potential_feature_set = current_set_of_features.copy()
            eliminated_feature = potential_feature_set.pop(k)

            accuracy = leave_one_out_cross_validation(data, potential_feature_set, None)
            print(f"Removing feature {eliminated_feature} accuracy is {accuracy:.1f}%")

            if accuracy > best_so_far_accuracy:      
                best_so_far_accuracy = accuracy
                feature_to_remove_at_this_level = eliminated_feature


        if feature_to_remove_at_this_level is not None:
            current_set_of_features.remove(feature_to_remove_at_this_level)

            if best_so_far_accuracy > best_accuracy:
                best_accuracy = best_so_far_accuracy
                best_feature_set = current_set_of_features.copy()
                print(f"Removing feature {feature_to_remove_at_this_level} was best, accuracy is {best_so_far_accuracy:.1f}%\n")
            else:
                print("Accuracy dropped! Continuing search...")

    return best_feature_set, best_accuracy 
        

if __name__ == "__main__":

    startAlg()