# Sample 2: Bare Except Clause Issue

def parse_number(val):
    try:
        result = int(val)
        return result
    except:
        # Bare except suppresses SystemExit, KeyboardInterrupt, etc.
        print("Failed to parse number")
        return None


def calculate_discount(price, rate):
    try:
        return price * (1 - rate)
    except:
        pass
