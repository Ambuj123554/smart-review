# Sample 7: Contextual & Logic Issues (For AI & AST Detection)

def find_adjacent_pairs(items):
    pairs = []
    # Logic Error / Off-by-one: items[i + 1] will raise IndexError on the last loop iteration
    for i in range(len(items)):
        if items[i] == items[i + 1]:
            pairs.append(items[i])
    return pairs


def process_scores(scores):
    total = sum(scores)
    # Potential ZeroDivisionError if scores is empty
    average = total / len(scores)
    
    if average == None: # AST-003: Compare to None with ==
        return 0
        
    return average
    return -1 # AST-004: Unreachable code


def check_status_code(code):
    if code is 200: # AST-005: Using 'is' for int literal
        return "OK"
    return "ERROR"
