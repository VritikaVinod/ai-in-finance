# StartupLens interactive investor dashboard

## Run on Windows
1. Extract the ZIP to a folder.
2. Open that folder in File Explorer.
3. Double-click `run_app.bat` (or run the commands below in PowerShell).

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The browser opens the local dashboard. Keep the terminal window open while using it.

## Included features
- Startup profile and funding trajectory input form
- Funding-year selection and derived timeline fields
- Logistic Regression as the default primary model
- Random Forest and Extra Trees as selectable alternatives
- Side-by-side model scores for the same startup
- One-feature-at-a-time local sensitivity explanation
- Cross-validation metric dashboard and confusion-matrix counts
- Explanations of the models, metrics, threshold and limitations
- CSV export of an assessment

## Evaluation transparency
The supplied model comparison is based on 4-fold stratified cross-validation with out-of-fold predictions and a fixed 0.50 threshold. It is not a separate untouched holdout test set. The dataset contains 169 companies and only 8 failures. The model outputs are not calibrated real-world probabilities.

## Important input note
The funding year controls calculate observed funding span, company age at first/last observed funding, and rounds per observed funding year. The remaining funding-trajectory values are entered as aggregate features. Use the feature definitions and units in the source workbook. If a field is unknown, leave the dataset-default value rather than inventing a value.

## Interpretation note
The explanation panel compares the entered value of one feature with a typical value from the training cohort while holding other inputs fixed. This is a local sensitivity explanation, not a causal explanation. Correlated features can affect it.
