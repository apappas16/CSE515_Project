


# FUNCTIONS:
# Takes in the results from the previous query and displays the results in a new different order
# given list of irrelevant and relevant gestures
def reorder_results(results, relevantList, irrelevantList):
    new_results = []
    last_results = []
    for gesture in results:
        if gesture in relevantList:
            new_results.append(gesture)
        elif gesture not in irrelevantList:
            last_results.append()
    new_results = new_results + last_results
    print("Re-ordered results:\n")
    for result in new_results:
        print(result)

# Changes some parameters from original query and returns new results as well as what was changed
# about the original query
def revise_query(results):
    pass


# prints out to the user a list of new results after either re-ordering results or revising query
def display_new_results(new_results):
    pass


# determines probability feedback from the list of results???
def prob_feedback():
    pass

# END OF FUNCTIONS


if __name__ == "__main__":
    results_file = open("phase3_query_results.txt", "r")
    all_results = results_file.readlines()
    num_all_results = len(all_results)

    relevant_results_file = open("relevant_results.txt", "r")
    relevant_results = relevant_results_file.readlines()
    num_relevant = len(relevant_results)

    irrelevant_results_file = open("irrelevant_results.txt", "r")
    irrelevant_results = irrelevant_results_file.readlines()
    num_irrelevant = len(irrelevant_results)

    # decide whether to re-order results or revise query
    # check for number of relevant/irrelevant results and decide that way?
    #       choose to reorder results if there are more relevant results
    if num_relevant > num_irrelevant:
        reorder_results(all_results, relevant_results, irrelevant_results)
    elif num_irrelevant > num_relevant:
        revise_query(all_results)
