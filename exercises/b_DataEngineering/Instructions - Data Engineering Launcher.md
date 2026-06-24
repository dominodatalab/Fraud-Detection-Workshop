# Data Engineering — Launcher

In this exercise we will re-run the data engineering pipeline using a Domino Launcher. Instead of hardcoding parameters in the script, a Launcher exposes a web form that lets users configure and trigger jobs without touching any code.

## One-Time Setup: Create the Launcher

> This step is completed once by the data scientist / instructor. Attendees will use the launcher in the Testing section below.

1. In your Domino project, navigate to **Deployments > Launchers**

2. Click **New Launcher**

3. Fill in the following fields:

   - **Name:** `Data Engineering Pipeline`
   - **Description:** `Parameterized data engineering job for the fraud detection pipeline.`
   - **Command:**
     ```
     python exercises/b_DataEngineering/data_engineering_launcher.py ${input_filename} ${output_filename} ${generate_eda_report}
     ```

4. Add the following parameters using **Add Parameter**:

   | Parameter Name | Title | Control Type | Default Value |
   |---|---|---|---|
   | `input_filename` | Input Dataset Filename | Text | `clean_cc_transactions.csv` |
   | `output_filename` | Output Dataset Filename | Text | `transformed_cc_transactions.csv` |
   | `generate_eda_report` | Generate EDA Report | Checkbox | Checked |

5. Click **Save**

---

## Testing the Launcher

### Test 1 — Default Run (Happy Path)

1. Navigate to **Deployments > Launchers** and click **Data Engineering Pipeline**

2. Leave all fields at their defaults and click **Run**

3. Navigate to **Jobs** in the left sidebar and confirm the job appears and completes successfully

4. Open the completed job and verify:
   - Artifacts tab contains `preprocessing_report.html`
   - Output log shows `saved to .../transformed_cc_transactions.csv`
   - MLflow experiment `CC Fraud Preprocessing` contains a new run with params `input_filename`, `output_filename`, and `generate_eda_report`

---

### Test 2 — Skip EDA Report

1. Open the Launcher again

2. **Uncheck** the `Generate EDA Report` checkbox

3. Click **Run**

4. Once complete, confirm:
   - Job finishes noticeably faster (EDA report generation is skipped)
   - Artifacts tab does **not** contain `preprocessing_report.html`
   - `transformed_cc_transactions.csv` is still saved correctly

---

### Test 3 — Custom Output Filename

1. Open the Launcher again

2. Change `Output Dataset Filename` to `transformed_cc_transactions_v2.csv`

3. Click **Run**

4. Once complete, confirm:
   - A new file `transformed_cc_transactions_v2.csv` exists in the Domino Dataset
   - The original `transformed_cc_transactions.csv` is unchanged

---

## New Domino Concepts

**Launchers:**
> Launchers are self-serve web forms that data scientists build to expose pre-configured jobs to non-technical users. When a user fills in the form and clicks Run, Domino passes the inputs into the script as command-line arguments and starts a job — no coding required. This makes it easy for business users, analysts, and operators to trigger pipelines safely and reproducibly.
