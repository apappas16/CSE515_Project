


# FUNCTIONS:
# Takes in the results from the previous query and displays the results in a new different order
# given list of irrelevant and relevant gestures
def reorder_results(results, relevantList, irrelevantList):
    new_results = []
    neutral_results = []
    last_results = []
    for gesture in results:
        if gesture in relevantList:
            new_results.append(gesture)
        elif gesture not in irrelevantList:
            neutral_results.append(gesture)
        elif gesture in irrelevantList:
            last_results.append(gesture)
    new_results = new_results + neutral_results + last_results
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
    results_file = open("similarGesturesTask6.txt", "r")
    all_results = results_file.read()
    all_results.strip("[")
    all_results.strip("(")
    all_results.strip("]")
    all_results.strip(")")
    all_results.split(", ")
    num_all_results = len(all_results) / 2

    relevant_results_file = open("relevent.txt", "r")
    relevant_results = relevant_results_file.read()
    relevant_results.strip("'")
    relevant_results.strip("[")
    relevant_results.strip("]")
    relevant_results.split(", ")
    num_relevant = len(relevant_results)

    irrelevant_results_file = open("irrelevent.txt", "r")
    irrelevant_results = irrelevant_results_file.read()
    irrelevant_results.strip("'")
    irrelevant_results.strip("[")
    irrelevant_results.strip("]")
    irrelevant_results.split(", ")
    num_irrelevant = len(irrelevant_results)

    # decide whether to re-order results or revise query
    # check for number of relevant/irrelevant results and decide that way?
    #       choose to reorder results if there are more relevant results
    if num_relevant > num_irrelevant:
        reorder_results(all_results, relevant_results, irrelevant_results)
    elif num_irrelevant > num_relevant:
        revise_query(all_results)
