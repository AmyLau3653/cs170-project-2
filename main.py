import numpy as np
import time


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
    
    # start run time
    start_time = time.time()

    # run the selected algorithm
    if choice == "1":
        best_features, best_accuracy  = forward_selection(data)
    # elif choice == "2":
    #     best_features = backward_elimination(data)
    # else:
    #     print("Invalid input. Please restart and select 1 or 2.")
    #     return
    
    # end run time
    end_time = time.time()
    # total run time
    run_time = end_time - start_time

    # print best features and accuracy
    print(f"Finished search!! The best feature subset is {best_features}, which has an accuracy of {best_accuracy:.1f}%")
    # print run time
    print(f"Algorithm runtime: {run_time:.2f} seconds")

# leave one out cross validation, based on matlab code provided by Professor Keogh in the Project 2 Briefing Slides
# function accuracy = leave_one_out_cross_validation(data, current_set, feature_to_add)
def leave_one_out_cross_validation(data, current_set, feature_to_add):
    num_correctly_classified = 0
    num_instances = data.shape[0]
    
    feature_set_to_use = current_set if feature_to_add is None else current_set + [feature_to_add]

    # for i = 1 : size(data,1)  % Loop through each instance
    for i in range(num_instances): 
        # object_to_classify = data(i,2:end);  % Extract feature values (ignoring class label)
        object_to_classify = data[i, feature_set_to_use]  # get features, ignore class label\
        # label_object_to_classify = data(i,1);  % Extract class label
        label_object_to_classify = data[i, 0]  # get class label
        
        # nearest_neighbor_distance = inf;  % Initialize min distance as infinity
        nearest_neighbor_distance = float('inf')
        # nearest_neighbor_location = inf;
        nearest_neighbor_label = None
        
        # for k = 1 : size(data,1)  % Loop through all other instances
        for k in range(num_instances): 
            # if k ~= i  % Exclude the test instance itself
            if k != i: 
                # distance = sqrt(sum((object_to_classify - data(k,2:end)).^2));  % Compute Euclidean distance
                # source: https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/
                distance = np.linalg.norm(object_to_classify - data[k, feature_set_to_use])  # get euclidean distance
                
                # if distance < nearest_neighbor_distance  % Update nearest neighbor
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    # nearest_neighbor_location = k;
                    # nearest_neighbor_label = data(nearest_neighbor_location,1);  % Get class of nearest neighbor
                    nearest_neighbor_label = data[k, 0]

        if label_object_to_classify == nearest_neighbor_label:
            # number_correctly_classfied = number_correctly_classfied + 1;  % Count correct predictions
            num_correctly_classified += 1

    # accuracy = number_correctly_classfied / size(data,1);  % Compute accuracy
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

    # for i = 1: size(data, 2) -1
    for i in range(num_features):
        # disp(['On the ', num2str(i), 'th level of the search tree'])
        print(f"On the {i+1}th level of the search tree")
        # feature_to_add_at_this_level = [];
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0

        # for k = 1: size(data, 2) - 1
        for k in range(1, num_features + 1):
            # if isempty(intersect(current_set_of_features, k))
            if k not in current_set_of_features:  # only consider features not already added
                # disp(['--Considering adding the ', num2str(k), ' feature'])
                print(f"--Considering adding feature {k}")
                # accuracy = leave_one_out_cross_validation(data, current_set_of_features, k+1);
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k)

                if accuracy > best_so_far_accuracy:      
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k


        if feature_to_add_at_this_level is not None:  # add the best feature found
            # current_set_of_features(i) = feature_to_add_at_this_level;
            current_set_of_features.append(feature_to_add_at_this_level)
            # disp(['On level ', num2str(i), ' I added feature ', num2str(feature_to_add_at_this_level), ' to current set'])
            print(f"On level {i+1}, added feature {feature_to_add_at_this_level} to current set: {current_set_of_features}")

            if best_so_far_accuracy > best_accuracy:  # update best accuracy so far
                best_accuracy = best_so_far_accuracy
                best_feature_set = current_set_of_features.copy()

    return best_feature_set, best_accuracy 
        

if __name__ == "__main__":

    startAlg()