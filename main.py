from data_process import *

if __name__ == '__main__':
    train_time = '2014-09-05'
    test_time = '2014-09-06'
    # Use data in the time range till train_time to train.
    have_train, pairs_train, transact_train, wish_train = read_data('./bm1y', end_time=train_time)
    # Get the test data.
    # have_test, pairs_test, transact_test, wish_test = read_data('./bm1y', end_time=test_time)
    have_test, pairs_test, transact_test, wish_test = read_data('./bm1y', start_time=train_time, end_time=test_time)
    # Whole data set.
    have, pairs, transact, wish = read_data('./bm1y', end_time=test_time)

    # Use whole dataset to construct the mapping relationship.
    users = get_unique_users(have, transact, wish, pairs)
    products = get_unique_products(have, transact, wish, pairs)

    # To get the mapping from users ID to index for simplicity, e.g. From user ID "44" to user "1"
    users_dict = {}
    for index, value in zip(range(len(users)), users):
        users_dict[value] = index

    products_dict = {}
    for index, value in zip(range(len(products)), products):
        products_dict[value] = index

    # The feature space rank.
    rank = 10  # TBD using some known methods.
    # The tensor we want to learn.
    # tensor = [have_train, wish_train, transact_train, pairs_train]
    tensor = [have_train, wish_train, transact_train]
    # First two steps in Algorithm 1 (Slice Learning): Farias and Li (2019)
    P_U, P_V = learning_feature(tensor, rank, users_dict, products_dict)
    # The 3rd step
    hat_M = P_U @ gen_interaction_matrix(transact_train, users_dict, products_dict) @ P_V

    transact_test_transform = gen_interaction_matrix(transact_test, users_dict, products_dict).toarray()
    theta_num = 1
    theta_set = [0.0001]
    # theta_set = np.linspace(0, 0.01, theta_num)
    curve = np.zeros((theta_num,2))
    for idx,theta in zip(range(theta_num),theta_set):
        print(theta)
        # curve[idx, 0] is TPR, curve[idx, 1] is FPR.
        curve[idx] = evaluate_predict(hat_M, transact_test_transform , theta)
    print(curve)