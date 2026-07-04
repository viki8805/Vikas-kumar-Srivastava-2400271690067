import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Smart Loan Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

model = joblib.load("models/loan_model.pkl")

# ---------------- CSS ---------------- #

st.markdown("""
<style>

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

.stApp{
background:linear-gradient(135deg,#EEF4FF,#DCEBFF,#F7FBFF);
}

.block-container{
padding-top:2rem;
padding-bottom:2rem;
}

.title{
font-size:42px;
font-weight:700;
color:#1E3A8A;
}

.subtitle{
font-size:18px;
color:#64748B;
margin-bottom:25px;
}

.card{
background:white;
padding:25px;
border-radius:20px;
box-shadow:0px 8px 20px rgba(0,0,0,.08);
margin-bottom:20px;
}

.dev{
background:linear-gradient(135deg,#2563EB,#1D4ED8);
padding:25px;
border-radius:20px;
color:white;
text-align:center;
box-shadow:0px 8px 20px rgba(0,0,0,.15);
}

.metric{
background:white;
padding:20px;
border-radius:18px;
text-align:center;
box-shadow:0px 6px 15px rgba(0,0,0,.08);
}

.metric h1{
color:#2563EB;
font-size:30px;
margin:0;
}

.metric h4{
color:gray;
margin:0;
}

.stButton>button{
width:100%;
height:60px;
border:none;
border-radius:15px;
font-size:20px;
font-weight:bold;
background:linear-gradient(90deg,#2563EB,#06B6D4);
color:white;
}

.stButton>button:hover{
background:linear-gradient(90deg,#1D4ED8,#0891B2);
}

</style>
""",unsafe_allow_html=True)

# ---------------- Header ---------------- #

st.markdown("""
<div class='title'>
🏦 Smart Loan Prediction System
</div>

<div class='subtitle'>
Decision Tree Based Machine Learning Application
</div>
""",unsafe_allow_html=True)





st.write("")

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("📝 Applicant Details")

age = st.number_input("Age", 18, 100, 25)

income = st.number_input("Annual Income (₹)", 0, 10000000, 50000)

loan = st.number_input("Loan Amount (₹)", 0, 5000000, 100000)

credit = st.slider("Credit Score", 300, 900, 700)

predict = st.button("🚀 Predict Loan Status")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Prediction ---------------- #

if predict:

    data = np.array([[age, income, loan, credit]])

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)

    st.write("")
    st.markdown("## 📋 Prediction Report")

    col1, col2 = st.columns([1.8,1])

    with col1:

        report = pd.DataFrame({
            "Field":[
                "Age",
                "Annual Income",
                "Loan Amount",
                "Credit Score"
            ],
            "Value":[
                age,
                f"₹ {income:,}",
                f"₹ {loan:,}",
                credit
            ]
        })

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("Applicant Summary")

        st.table(report)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("Prediction")

        if prediction == 1:

            st.success("🎉 Loan Approved")

            st.balloons()

            prob = probability[0][1] * 100

            st.metric(
                "Approval Chance",
                f"{prob:.2f}%"
            )

            st.progress(min(int(prob),100))

        else:

            st.error("❌ Loan Rejected")

            prob = probability[0][0] * 100

            st.metric(
                "Rejection Chance",
                f"{prob:.2f}%"
            )

            st.progress(min(int(prob),100))

        st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    chart1, chart2 = st.columns(2)

    with chart1:

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("💰 Income vs Loan")

        df_chart = pd.DataFrame({
            "Category": ["Income", "Loan Amount"],
            "Amount": [income, loan]
        })

        fig = px.bar(
            df_chart,
            x="Category",
            y="Amount",
            color="Category",
            text="Amount",
            color_discrete_sequence=["#2563EB", "#06B6D4"]
        )

        fig.update_layout(
            height=380,
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with chart2:

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("⭐ Credit Score")

        fig2 = px.pie(
            names=["Credit Score", "Remaining"],
            values=[credit, 900-credit],
            hole=.70,
            color_discrete_sequence=["#2563EB", "#E5E7EB"]
        )

        fig2.update_layout(
            height=380,
            showlegend=True,
            margin=dict(l=10, r=10, t=20, b=10)
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("💡 Smart Recommendation")

    if prediction == 1:

        st.success("""
### 🎉 Congratulations!

Your profile looks suitable for loan approval.

✔ Good Credit Score

✔ Strong Financial Profile

✔ Decision Tree predicts approval.
""")

    else:

        st.warning("""
### Improve Your Profile

• Increase Annual Income

• Improve Credit Score

• Reduce Loan Amount

• Clear Existing EMI/Debt
""")

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

st.divider()

footer1, footer2, footer3 = st.columns(3)

with footer1:
    st.info("🏦 Smart Loan Prediction")

with footer2:
    st.success("⚡ Powered by Streamlit")

with footer3:
    st.warning("🤖 Decision Tree Classifier")

st.markdown(
"""
<div style='text-align:center;
padding:18px;
margin-top:20px;
color:gray;'>

<h3 style='color:#2563EB;'>
👨‍💻 Developed by
</h3>

<h2>
Vikas Kumar Srivastava
</h2>

<b>Roll No. : 2400271690067</b>

<br><br>

© 2026 Loan Prediction System

</div>
""",
unsafe_allow_html=True
)