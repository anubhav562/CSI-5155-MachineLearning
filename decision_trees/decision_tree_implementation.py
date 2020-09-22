import pandas as pd


class TreeNode:
    def __init__(self, gini_index, samples, feature_to_split_on, category_to_choose, left_child, right_child):
        self.gini_index = gini_index
        self.samples = samples
        self.feature_to_split_on = feature_to_split_on
        self.category_to_choose = category_to_choose
        self.left_child = left_child
        self.right_child = right_child


class DecisionTree:

    def __init__(self, criterion, max_depth=3):
        self.criterion = criterion
        self.label_column_name = None
        self.x_train_columns = None
        self.max_depth = max_depth
        self.features_category_pair_covered = []

    def calculate_gini_impurity(self, dataset):
        unique_labels_list = list(dataset[self.label_column_name].unique())
        probability_sq_summation = 0

        for label in unique_labels_list:
            probability_sq_summation += (len(dataset[dataset[self.label_column_name] == label])/len(dataset))**2

        gini_impurity = 1 - probability_sq_summation

        return gini_impurity

    def calculate_categorical_gini_index(self, dataset, feature, category):
        mask = dataset[feature] == category
        categorical_gini_impurity = (len(dataset[mask])/len(dataset))*self.calculate_gini_impurity(dataset[mask]) + \
            (len(dataset[~mask])/len(dataset))*self.calculate_gini_impurity(dataset[~mask])
        return categorical_gini_impurity

    def fit(self, x_train, y_train, depth=0):

        df = pd.concat([x_train, y_train], axis=1)

        self.label_column_name = y_train.columns[0]
        self.x_train_columns = list(x_train.columns)

        if depth == self.max_depth:
            return TreeNode(
                gini_index=self.calculate_gini_impurity(df), samples=len(df), feature_to_split_on=None,
                category_to_choose=None, left_child=None, right_child=None
            )

        if sum(list(x_train.describe().T["unique"])) == len(list(x_train.columns)):
            return TreeNode(
                gini_index=self.calculate_gini_impurity(df), samples=len(df), feature_to_split_on=None,
                category_to_choose=None, left_child=None, right_child=None
            )

        initial_gini_impurity = self.calculate_gini_impurity(dataset=df)

        feature_wise_dict_list = []

        for feature in x_train.columns:
            category_values = list(x_train[feature].unique())
            gini_scores = []
            for category in category_values:
                if (feature, category) in self.features_category_pair_covered:
                    continue
                temp_gini = self.calculate_categorical_gini_index(dataset=df, feature=feature, category=category)
                gini_scores.append(temp_gini)

            if len(gini_scores) == 0:
                continue

            minimum_gini_score = min(gini_scores)
            category_to_split_on = category_values[gini_scores.index(minimum_gini_score)]
            feature_wise_dict_list.append(
                {"feature": feature, "category": category_to_split_on, "gini_index": minimum_gini_score}
                 )

        if len(feature_wise_dict_list) == 0:
            return None

        feature_wise_gini_scores = [x["gini_index"] for x in feature_wise_dict_list]
        feature_to_split_on = feature_wise_dict_list[feature_wise_gini_scores.index(min(feature_wise_gini_scores))]

        self.features_category_pair_covered.append((feature_to_split_on["feature"], feature_to_split_on["category"]))

        node = TreeNode(
            gini_index=feature_to_split_on["gini_index"], samples=len(df),
            feature_to_split_on=feature_to_split_on["feature"],
            category_to_choose=feature_to_split_on["category"], left_child=None, right_child=None
        )

        mask = df[feature_to_split_on["feature"]] == feature_to_split_on["category"]

        left_child = self.fit(df[self.x_train_columns][mask], df[[self.label_column_name]][mask], depth+1)
        right_child = self.fit(df[self.x_train_columns][~mask], df[[self.label_column_name]][~mask], depth+1)

        node.left_child = left_child
        node.right_child = right_child

        return node


weather_df = pd.read_csv("weather.csv")

x_train_df = weather_df[["Weather", "Humidity"]]
y_train_df = weather_df[["Decision"]]

classifier = DecisionTree("gini")
root_node = classifier.fit(x_train_df, y_train_df)
