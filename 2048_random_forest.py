import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

moves_file_path = 'moves.csv'

moves_data = pd.read_csv(moves_file_path)
moves_data.columns = ['11', '12', '13', '14', '21', '22', '23', '24', '31', '32', '33', '34', '41', '42', '43', '44', 'move']
y = moves_data.move
features = ['11', '12', '13', '14', '21', '22', '23', '24', '31', '32', '33', '34', '41', '42', '43', '44']
X = moves_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
moves_model = DecisionTreeRegressor(random_state=1)
# Fit Model
moves_model.fit(train_X, train_y)


def getWrongValues(results, validation_values):
    result = 0
    for i in range(len(results)):
        if results[i] != validation_values.iloc[i]:
            result += 1
    return result


# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_predictions = list(map(lambda x: int(round(x)), rf_val_predictions))
error = getWrongValues(rf_val_predictions, val_y)/len(val_y)


print("error: ", error)
