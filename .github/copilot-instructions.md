# AI Coding Agent Instructions - Vehicle Price Predictor

## Project Overview

This is a **Sri Lanka vehicle price prediction application** using XGBoost. The system predicts vehicle prices from the riyasewana.com dataset based on vehicle specifications. It's deployed as a Streamlit web app with an ML model backend.

## Architecture & Data Flow

### Phase 1: Data Scraping & Preprocessing
- **scraper-riyasewana.py** - Web scraper (not actively used; data already in `riyasewana_raw_data.csv`)
- **pre-processing-data.py** - Cleans raw data and engineers features:
  - Removes invalid prices (< 500k rental data, > 80M outliers)
  - Fills missing mileage with median values
  - Creates `Vehicle_Age` from Year of Manufacture (current year: 2026)
  - Extracts vehicle features from Options string (Has_AC, Has_PowerSteering, Has_PowerMirror)
  - One-hot encodes Transmission, Fuel_Type, and Make
  - Output: `riyasewana_cleaned.csv`

### Phase 2: Model Training
- **train.py** - Trains XGBoost with GridSearchCV:
  - Target is **log-transformed** (y_log = log1p(price)) to handle wide price range (500k-80M)
  - Baseline: Linear Regression (for comparison)
  - Advanced: XGBoost with hyperparameter tuning
  - Saves: `xgboost_best_model.pkl`, test data (`X_test.csv`, `y_test.csv`)
  - Key insight: Model trained on log scale, predictions converted back with `expm1()`

### Phase 3: Web Application
- **app.py** (at project root, not in subdirectory) - Streamlit UI:
  - Loads pretrained model from `lightgbm_best_model.pkl` (or `xgboost_best_model.pkl` depending on version)
  - Interactive inputs: Make, Model frequency, Year, Mileage, Engine, Transmission, Fuel, Features
  - Displays price prediction with custom CSS styling
  - Theme-adaptive colors using Streamlit CSS variables
  - Prediction confidence gauge using Plotly
  - Price range calculator (±10% variance)

### Phase 4: Explainability (XAI)
- **xai.py** - SHAP analysis for model interpretability:
  - Uses TreeExplainer for XGBoost model
  - Generates feature importance visualizations
  - Explains individual predictions
  - Outputs: `shap_summary.png`, `shap_individual_explanation.png`

## Critical Developer Workflows

### Run Full Pipeline
```bash
python pre-processing-data.py  # Clean raw data
python train.py                # Train model (generates ~139 lines of output)
streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
```

### Development Server
```bash
streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
# App runs on http://localhost:8501
```

### Generate Model Explanations
```bash
python xai.py  # Requires trained model & test data
```

### Deployment to Streamlit Cloud
1. Push to GitHub: `git add . && git commit -m "Deploy" && git push`
2. Go to https://streamlit.io/cloud → Select repository → `chanisagithub/assignment-05`
3. **Main file path**: `app.py` (at root level, not subdirectory)
4. **Branch**: `main` or `development`
5. Streamlit will auto-detect `requirements.txt` and install dependencies
6. ⚠️ **Do NOT use Vercel** - Streamlit requires persistent WebSocket, Vercel is serverless
- See [DEPLOYMENT.md](DEPLOYMENT.md) for alternative platforms (Heroku, Railway, Render)

## Project-Specific Patterns & Conventions

### Data Handling
- **Price range outlier removal** (lines 17-18 in train.py): `df = df[(df['Price'] > 500000) & (df['Price'] < 80000000)]`
- **Median filling** used consistently for missing values, not mean
- **Log transformation** mandatory for price predictions - always convert with `np.log1p()` and revert with `np.expm1()`

### Feature Engineering
- **Model_Freq**: Maps vehicle model to frequency count instead of categorical encoding (preserves rarity signal)
- **Options parsing**: Extract boolean features from concatenated string column using `str.contains()`
- **Vehicle_Age**: Always calculated as `2026 - YOM`, not current year (hardcoded assumption)

### Model Training
- GridSearchCV used for hyperparameter tuning (not manual parameter search)
- Baseline Linear Regression included for comparison
- Test data (20%) saved separately for SHAP analysis - do not use for training validation
- Random state: 42 (for reproducibility)

### Streamlit App Conventions
- Page config: Wide layout, collapsed sidebar, custom icon "🚗"
- CSS styling uses CSS variables: `var(--background-color)`, `var(--text-color)`, `var(--secondary-background-color)`
- Gradient button styling: Linear gradient from `#667eea` to `#764ba2`
- All model loading uses `joblib.load()` not pickle directly

## Integration Points & Dependencies

### External Data Sources
- Dataset: riyasewana.com (Sri Lankan vehicle marketplace)
- Current implementation uses static CSVs, not live API

### Key Dependencies (from requirements.txt)
- **streamlit** - Web UI framework
- **xgboost** - ML model (not sklearn)
- **scikit-learn** - Preprocessing (train_test_split, encoding)
- **joblib** - Model serialization
- **pandas/numpy** - Data processing
- **plotly** - Interactive visualizations
- **shap** - Model explanability (optional, for xai.py)

### File Dependencies
- `app.py` requires: `lightgbm_best_model.pkl`, `requirements.txt`
- `train.py` requires: `riyasewana_cleaned.csv`
- `pre-processing-data.py` requires: `riyasewana_raw_data.csv`
- **All at project root for Streamlit Cloud** (not in subdirectories)

## Common Pitfalls to Avoid

1. **Don't forget log transformation**: Always train on `log1p(price)`, display predictions with `expm1(log_prediction)`
2. **Year assumption**: Vehicle age uses hardcoded 2026, not `datetime.now().year`
3. **Model serialization**: Use joblib, not pickle directly
4. **Deployment platform**: Vercel won't work (no persistent server) - use Streamlit Cloud, Heroku, or Railway
5. **Test data isolation**: X_test.csv must not be used during training - it's reserved for SHAP analysis

## Key Files Reference

| File | Purpose | Modifiable |
|------|---------|-----------|
| [app.py](app.py) | Web interface | Yes (UI/features) |
| [train.py](train.py) | Model training | Yes (hyperparameters) |
| [pre-processing-data.py](pre-processing-data.py) | Data cleaning | Yes (feature engineering) |
| [xai.py](xai.py) | Model explainability | Yes (visualization only) |
| [requirements.txt](requirements.txt) | Dependencies | Yes (versions) |
| `xgboost_best_model.pkl` | Trained model | No (binary artifact) |
| `riyasewana_cleaned.csv` | Training data | No (generated) |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Deployment guide | Reference only |
