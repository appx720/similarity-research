import itertools
from algorithm import main


if __name__ == "__main__":
    users_list = [100]
    items_list = [100, 200, 500, 1000, 5000, 10000, 100000]

    for users, items in itertools.product(users_list, items_list):
        main(users, items)