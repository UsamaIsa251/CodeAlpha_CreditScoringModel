import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="💳",
    layout="wide",
)

MODEL_PATH = "outputs/best_credit_scoring_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("💳 Credit Scoring System")
st.write(
    "Estimate credit risk using a machine-learning model "
    "trained on the German Credit dataset."
)

st.info(
    "This is an educational demonstration, not a lending decision. "
    "Do not enter real personal or financial information."
)

with st.form("credit_form"):
    st.subheader("Applicant Financial Information")

    col1, col2 = st.columns(2)

    with col1:
        checking_account = st.selectbox(
            "Checking account status",
            ["A11", "A12", "A13", "A14"],
            format_func=lambda x: {
                "A11": "Below 0 DM",
                "A12": "0 to 200 DM",
                "A13": "200 DM or more",
                "A14": "No checking account",
            }[x],
        )

        duration = st.number_input(
            "Loan duration (months)", min_value=1, max_value=72, value=12
        )

        credit_history = st.selectbox(
            "Credit history",
            ["A30", "A31", "A32", "A33", "A34"],
        )

        purpose = st.selectbox(
            "Loan purpose",
            [
                "A40", "A41", "A42", "A43", "A44",
                "A45", "A46", "A47", "A48", "A49", "A410",
            ],
        )

        credit_amount = st.number_input(
            "Credit amount (DM)", min_value=1, value=2000
        )

        savings_account = st.selectbox(
            "Savings account status",
            ["A61", "A62", "A63", "A64", "A65"],
        )

        employment_since = st.selectbox(
            "Employment duration",
            ["A71", "A72", "A73", "A74", "A75"],
            format_func=lambda x: {
                "A71": "Unemployed",
                "A72": "Less than 1 year",
                "A73": "1 to 4 years",
                "A74": "4 to 7 years",
                "A75": "7 years or more",
            }[x],
        )

        installment_rate = st.selectbox(
            "Installment rate (% of disposable income)",
            [1, 2, 3, 4],
        )

        personal_status_sex = st.selectbox(
            "Personal status and sex",
            ["A91", "A92", "A93", "A94", "A95"],
        )

        other_debtors = st.selectbox(
            "Other debtors / guarantors",
            ["A101", "A102", "A103"],
        )

    with col2:
        residence_since = st.selectbox(
            "Years at current residence", [1, 2, 3, 4]
        )

        property_type = st.selectbox(
            "Property",
            ["A121", "A122", "A123", "A124"],
        )

        age = st.number_input(
            "Age (years)", min_value=18, max_value=100, value=35
        )

        other_installment_plans = st.selectbox(
            "Other installment plans",
            ["A141", "A142", "A143"],
        )

        housing = st.selectbox(
            "Housing",
            ["A151", "A152", "A153"],
        )

        existing_credits = st.selectbox(
            "Number of existing credits",
            [1, 2, 3, 4],
        )

        job = st.selectbox(
            "Job category",
            ["A171", "A172", "A173", "A174"],
        )

        dependents = st.selectbox(
            "Number of dependents", [1, 2]
        )

        telephone = st.selectbox(
            "Telephone",
            ["A191", "A192"],
            format_func=lambda x: {
                "A191": "No telephone",
                "A192": "Telephone registered",
            }[x],
        )

        foreign_worker = st.selectbox(
            "Foreign worker",
            ["A201", "A202"],
            format_func=lambda x: {
                "A201": "Yes",
                "A202": "No",
            }[x],
        )

    submitted = st.form_submit_button(
        "Assess Credit Risk",
        type="primary",
        use_container_width=True,
    )

if submitted:
    applicant = pd.DataFrame([{
        "checking_account": checking_account,
        "duration": duration,
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": credit_amount,
        "savings_account": savings_account,
        "employment_since": employment_since,
        "installment_rate": installment_rate,
        "personal_status_sex": personal_status_sex,
        "other_debtors": other_debtors,
        "residence_since": residence_since,
        "property": property_type,
        "age": age,
        "other_installment_plans": other_installment_plans,
        "housing": housing,
        "existing_credits": existing_credits,
        "job": job,
        "dependents": dependents,
        "telephone": telephone,
        "foreign_worker": foreign_worker,
    }])

    try:
        prediction = int(model.predict(applicant)[0])
        probability = float(model.predict_proba(applicant)[0, 1])

        st.subheader("Assessment Result")
        st.metric("Estimated bad-credit probability", f"{probability:.1%}")

        if prediction == 1:
            st.warning(
                "The model predicts the bad-credit class. "
                "This is an estimate, not a definitive judgment."
            )
        else:
            st.success(
                "The model predicts the good-credit class. "
                "This is an estimate, not a guarantee of repayment."
            )

        st.caption(
            "Model predictions may be inaccurate and should not be used "
            "as the sole basis for real financial decisions."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
