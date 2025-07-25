
from predictor_module import HouseData, FeatureScaler, HousePriceModel, ModelEvaluator
from sklearn.model_selection import train_test_split

# Step 1: Load dataset
file_path = "housing.csv"  # Or full path if needed
data = HouseData(file_path)
X, y = data.load_data()

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Scale features
scaler = FeatureScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train model
model = HousePriceModel()
model.train(X_train_scaled, y_train)

# Step 5: Predict
y_pred = model.predict(X_test_scaled)

# Step 6: Evaluate
evaluator = ModelEvaluator()
evaluator.evaluate(y_test, y_pred)
