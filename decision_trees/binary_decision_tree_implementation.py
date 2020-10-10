import pandas as pd


class TreeNode:
    def __init__(self, split=None, target_class=None,  left_child=None, right_child=None, value_counts=None):
        self.split = split
        self.left_child = left_child
        self.right_child = right_child
        self.target_class = target_class
        self.value_counts = value_counts


class DecisionTree:
    def __init__(self):
        self.target_variable = None
        self.split_condition_bucket = {}

    def _calculate_gini_index(self, dataset):
        class_values = list(dataset[self.target_variable].unique())
        p_square_summation = 0

        for class_value in class_values:
            p_square_summation += (len(dataset[dataset[self.target_variable]==class_value])/len(dataset))**2

        gini_index = 1 - p_square_summation

        return gini_index

    def _calculate_gini_index_for_split_test(self, dataset, feature, feature_value):
        d_t = dataset[dataset[feature] == feature_value]
        d_not_t = dataset[dataset[feature] != feature_value]

        d_t_proportion = len(d_t)/len(dataset)
        d_not_t_proportion = 1 - d_t_proportion

        gini_d_t = self._calculate_gini_index(dataset=d_t)
        gini_d_not_t = self._calculate_gini_index(dataset=d_not_t)

        gini_index_for_split = d_t_proportion*gini_d_t + d_not_t_proportion*gini_d_not_t

        return gini_index_for_split

    def _is_dataset_homogeneous(self, dataset):
        return True if len(dataset[self.target_variable].unique()) == 1 else False

    def _is_more_splittable(self, dataset):
        columns = [column for column in dataset.columns if column != self.target_variable]

        for column in columns:
            category_values = list(dataset[column].unique())
            if len(category_values) != 1:
                return True

        return False

    def _find_best_split_and_divide_data(self, dataset, features, t_node):
        gini_index = 1
        feature_to_split_on = None
        category_to_split_on = None

        for feature in features:
            category_values = list(dataset[feature].unique())

            if len(category_values) == 1:
                continue

            for category in category_values:

                if self.split_condition_bucket.get(f"{feature}={category}"):
                    continue

                gini_for_split = self._calculate_gini_index_for_split_test(dataset, feature, category)

                if gini_index > gini_for_split:
                    gini_index = gini_for_split
                    feature_to_split_on = feature
                    category_to_split_on = category

        t_node.split = f"{feature_to_split_on}={category_to_split_on}"

        print(t_node.split)

        self.split_condition_bucket[f"{feature_to_split_on}={category_to_split_on}"] = 1

        left_dataset = dataset[dataset[feature_to_split_on] == category_to_split_on]
        right_dataset = dataset[dataset[feature_to_split_on] != category_to_split_on]

        return left_dataset, right_dataset

    def grow_tree(self, dataset, features, target_variable):
        self.target_variable = target_variable

        if self._is_dataset_homogeneous(dataset):
            if dataset.empty:
                return TreeNode()
            return TreeNode(
                target_class=dataset[self.target_variable].iloc[0],
                value_counts=dict(dataset[self.target_variable].value_counts())
            )

        if not self._is_more_splittable(dataset):
            value_counts = dict(dataset[self.target_variable].value_counts())
            return TreeNode(value_counts=value_counts)

        t_node = TreeNode()
        left_dataset, right_dataset = self._find_best_split_and_divide_data(dataset, features, t_node)

        t_node.left_child = self.grow_tree(left_dataset, features, target_variable)
        t_node.right_child = self.grow_tree(right_dataset, features, target_variable)

        self.split_condition_bucket.pop(t_node.split)

        return t_node


df = pd.read_csv("weather.csv")

dt = DecisionTree()
root_node = dt.grow_tree(df, features=["Weather", "Humidity"], target_variable="Decision")
