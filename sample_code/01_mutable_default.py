# Sample 1: Mutable Default Argument Issue

def add_item(item, items=[]):
    """Appends item to list with mutable default argument."""
    items.append(item)
    return items


def create_user_profile(username, details={}):
    """Creates user profile dict with mutable default dictionary."""
    details["username"] = username
    return details


if __name__ == "__main__":
    list1 = add_item("apple")
    list2 = add_item("banana")
    print("List 2 output (unexpected accumulation):", list2)
