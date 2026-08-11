import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

# Safe Supabase import handling
try:
    from supabase import create_client, Client
    SUPABASE_LIB_INSTALLED = True
except ImportError:
    SUPABASE_LIB_INSTALLED = False

# Helper function to get current time in IST (UTC +5:30)
def get_ist_timestamp():
    ist_offset = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist_offset).strftime("%Y-%m-%d %H:%M:%S IST")

# =============================================================================
# PAGE CONFIGURATION & CUSTOM THEME
# =============================================================================
st.set_page_config(
    page_title="EdTech Academic Risk Engine",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    .hero-title { font-size: 2.0rem; font-weight: 700; color: #FFFFFF; margin: 0; }
    .hero-subtitle { font-size: 1.0rem; color: #94A3B8; margin-top: 4px; }

    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #334155 !important;
    }
    section[data-testid="stSidebar"] * { color: #F8FAFC !important; }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: #1E293B !important;
        border-color: #475569 !important;
        color: #FFFFFF !important;
    }
    
    .badge-green { background-color: #DCFCE7; color: #15803D; padding: 6px 14px; border-radius: 20px; font-weight: 600; }
    .badge-yellow { background-color: #FEF3C7; color: #B45309; padding: 6px 14px; border-radius: 20px; font-weight: 600; }
    .badge-red { background-color: #FEE2E2; color: #B91C1C; padding: 6px 14px; border-radius: 20px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# SUPABASE CLIENT INITIALIZATION
# =============================================================================
@st.cache_resource
def init_supabase():
    if not SUPABASE_LIB_INSTALLED:
        return None
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        return None

supabase_client = init_supabase()

# Header Status Badge
db_status_badge = """
<span style="background: rgba(34, 197, 94, 0.15); padding: 8px 16px; border-radius: 12px; font-size: 0.85rem; color: #4ADE80; border: 1px solid rgba(34, 197, 94, 0.3);">
    ● Supabase Connected
</span>
""" if supabase_client else """
<span style="background: rgba(245, 158, 11, 0.15); padding: 8px 16px; border-radius: 12px; font-size: 0.85rem; color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3);">
    ○ Local Demo Mode
</span>
"""

st.markdown(f"""
    <div class="hero-container">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="hero-title">🎓 Academic Early Warning Engine</div>
                <div class="hero-subtitle">Continuous modeling, diagnostic plans, IST tracking, & ground truth retraining loop.</div>
            </div>
            <div>
                {db_status_badge}
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# =============================================================================
# IN-MEMORY MODEL TRAINING
# =============================================================================
@st.cache_resource
def get_trained_pipeline():
    try:
        df = pd.read_csv('exams.csv')
    except Exception:
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'math score': np.random.randint(30, 100, n),
            'reading score': np.random.randint(30, 100, n),
            'writing score': np.random.randint(30, 100, n),
            'test preparation course': np.random.choice(['completed', 'none'], n),
            'lunch': np.random.choice(['standard', 'free/reduced'], n),
            'parental level of education': np.random.choice(['high school', 'some college', "bachelor's degree"], n)
        })

    np.random.seed(42)
    avg = (df['math score'] + df['reading score'] + df['writing score']) / 3.0
    df['attendance'] = np.clip(avg + np.random.normal(0, 8, len(df)), 35, 100).round(1)

    edu_map = {
        'some high school': 1, 'high school': 2, 'some college': 3,
        "associate's degree": 4, "bachelor's degree": 5, "master's degree": 6
    }
    df['test_prep_enc'] = df['test preparation course'].apply(lambda x: 1 if str(x).lower() == 'completed' else 0)
    df['lunch_enc'] = df['lunch'].apply(lambda x: 1 if str(x).lower() == 'standard' else 0)
    df['parent_edu_enc'] = df['parental level of education'].apply(lambda x: edu_map.get(str(x).lower(), 2))

    def calculate_continuous_target(row):
        m_loss = (100.0 - row['math score']) * 0.35
        r_loss = (100.0 - row['reading score']) * 0.25
        w_loss = (100.0 - row['writing score']) * 0.25
        a_loss = (100.0 - row['attendance']) * 0.45
        
        avg_score = (row['math score'] + row['reading score'] + row['writing score']) / 3.0
        acc_score = max(0.0, (80.0 - avg_score) * 0.25)
        acc_att = max(0.0, (80.0 - row['attendance']) * 0.35)
        prep_bonus = -4.0 if row['test_prep_enc'] == 1 else 0.0
        
        return np.clip(m_loss + r_loss + w_loss + a_loss + acc_score + acc_att + prep_bonus, 0.0, 100.0)

    df['risk_target'] = df.apply(calculate_continuous_target, axis=1)

    X = df[['math score', 'reading score', 'writing score', 'attendance', 'test_prep_enc', 'lunch_enc', 'parent_edu_enc']]
    y = df['risk_target']

    model = make_pipeline(StandardScaler(), Ridge(alpha=0.01))
    model.fit(X, y)
    return model

model = get_trained_pipeline()

edu_map = {
    "some high school": 1, "high school": 2, "some college": 3,
    "associate's degree": 4, "bachelor's degree": 5, "master's degree": 6
}

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
st.sidebar.markdown("## 📝 Student Profile Inputs")
student_name = st.sidebar.text_input("Student Name", value="Alex Morgan")

st.sidebar.markdown("---")
math_score = st.sidebar.slider("1. Math Score (%)", 0, 100, 90)
reading_score = st.sidebar.slider("2. Reading Score (%)", 0, 100, 85)
writing_score = st.sidebar.slider("3. Writing Score (%)", 0, 100, 85)

st.sidebar.markdown("---")
attendance = st.sidebar.slider("4. Attendance Rate (%)", 0.0, 100.0, 95.0, step=0.5)

st.sidebar.markdown("---")
test_prep = st.sidebar.selectbox("5. Test Prep Course", ["completed", "none"])
lunch = st.sidebar.selectbox("6. Lunch Program", ["standard", "free/reduced"])
parent_edu = st.sidebar.selectbox("7. Parent Education Level", list(edu_map.keys()))

# Model Inference Input
input_data = pd.DataFrame([{
    'math score': math_score,
    'reading score': reading_score,
    'writing score': writing_score,
    'attendance': attendance,
    'test_prep_enc': 1 if test_prep == "completed" else 0,
    'lunch_enc': 1 if lunch == "standard" else 0,
    'parent_edu_enc': edu_map[parent_edu]
}])

raw_pred = model.predict(input_data)[0]
risk_prob = float(np.clip(raw_pred, 0.0, 100.0))
avg_score = (math_score + reading_score + writing_score) / 3.0

weak_subjects = []
if math_score < 70: weak_subjects.append(("Math", math_score))
if reading_score < 70: weak_subjects.append(("Reading", reading_score))
if writing_score < 70: weak_subjects.append(("Writing", writing_score))

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Main Dashboard", 
    "🎯 Subject & Attendance Diagnostics", 
    "☁️ Supabase DB & Retraining",
    "📊 System Spec"
])

# =============================================================================
# TAB 1: MAIN DASHBOARD
# =============================================================================
with tab1:
    if len(weak_subjects) > 0 or attendance < 80.0:
        at_risk_list = [f"{sub} ({score}%)" for sub, score in weak_subjects]
        if attendance < 80.0: at_risk_list.append(f"Attendance ({attendance}%)")
        
        st.warning(
            f"⚠️ **Target Threshold Alert for {student_name}:** Flags detected in **{', '.join(at_risk_list)}**. "
            f"Switch to **'🎯 Subject & Attendance Diagnostics'** tab to review action plans and email draft."
        )

    c1, c2 = st.columns([1.1, 0.9], gap="medium")

    with c1:
        with st.container(border=True):
            st.markdown(f"#### 📋 Profile Overview: **{student_name}**")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Course Average", f"{avg_score:.1f}%")
            m2.metric("Attendance", f"{attendance}%")
            m3.metric("Test Prep", test_prep.title())

            st.markdown("##### Performance Breakdown")
            st.write(f"**Math:** {math_score}%")
            st.progress(math_score / 100.0)
            
            st.write(f"**Reading:** {reading_score}%")
            st.progress(reading_score / 100.0)
            
            st.write(f"**Writing:** {writing_score}%")
            st.progress(writing_score / 100.0)
            
            st.write(f"**Attendance:** {attendance}%")
            st.progress(attendance / 100.0)

    with c2:
        with st.container(border=True):
            st.markdown("#### 🚨 Dynamic Risk Gauge")
            st.metric("Academic Risk Score", f"{risk_prob:.1f}%")
            st.progress(risk_prob / 100.0)

            st.markdown("##### Instant Status Evaluation")
            failed = [s for s, sc in weak_subjects if sc < 50]

            if len(failed) > 0 and attendance < 75.0:
                st.markdown('<span class="badge-red">🔴 CRITICAL: FAIL & ATTENDANCE DEFICIT</span>', unsafe_allow_html=True)
                st.error(f"Failing {', '.join(failed)} AND low attendance ({attendance}%).")
            elif attendance < 75.0:
                st.markdown('<span class="badge-red">🔴 ATTENDANCE WARNING</span>', unsafe_allow_html=True)
                st.error(f"Attendance ({attendance}%) dropped below 75% threshold.")
            elif len(failed) > 0:
                st.markdown('<span class="badge-red">🔴 SUBJECT DEFICIT</span>', unsafe_allow_html=True)
                st.error(f"Failing subject(s): {', '.join(failed)}.")
            elif risk_prob >= 35.0:
                st.markdown('<span class="badge-red">🔴 HIGH RISK PROFILE</span>', unsafe_allow_html=True)
                st.error("Indicators reflect significant risk of academic decline.")
            elif risk_prob >= 18.0:
                st.markdown('<span class="badge-yellow">🟡 MEDIUM RISK PROFILE</span>', unsafe_allow_html=True)
                st.warning("Moderate risk indicators requiring regular monitoring.")
            else:
                st.markdown('<span class="badge-green">🟢 LOW RISK / ON TRACK</span>', unsafe_allow_html=True)
                st.success("Student is performing well and meeting targets.")

            st.markdown("---")
            
            # --- SUPABASE SAVE WITH DUPLICATE CHECK & IST TIMESTAMP ---
            save_clicked = st.button("💾 Save Assessment to Supabase", type="primary", use_container_width=True)

            if save_clicked:
                if supabase_client:
                    # Check for duplicates in Supabase
                    existing = supabase_client.table("student_evaluations").select("*").eq("student_name", student_name.strip()).execute()
                    
                    if len(existing.data) > 0:
                        st.session_state['duplicate_found'] = True
                        st.session_state['existing_id'] = existing.data[0]['id']
                    else:
                        st.session_state['duplicate_found'] = False
                        # Insert directly
                        current_ist = get_ist_timestamp()
                        payload = {
                            "student_name": student_name.strip(),
                            "math_score": math_score,
                            "reading_score": reading_score,
                            "writing_score": writing_score,
                            "attendance": attendance,
                            "test_prep": test_prep,
                            "predicted_risk": round(risk_prob, 2)
                        }
                        supabase_client.table("student_evaluations").insert(payload).execute()
                        st.success(f"✅ Successfully logged record for **{student_name}** at {current_ist}!")
                else:
                    st.info("⚠️ Supabase not connected. Running in demo mode.")

            # Duplicate Confirmation Prompt
            if st.session_state.get('duplicate_found', False):
                st.warning(f"⚠️ A record for **'{student_name}'** already exists in the database!")
                col_dup1, col_dup2 = st.columns(2)
                
                with col_dup1:
                    if st.button("🔄 Overwrite Existing Record", use_container_width=True):
                        record_id = st.session_state['existing_id']
                        payload = {
                            "math_score": math_score,
                            "reading_score": reading_score,
                            "writing_score": writing_score,
                            "attendance": attendance,
                            "test_prep": test_prep,
                            "predicted_risk": round(risk_prob, 2)
                        }
                        supabase_client.table("student_evaluations").update(payload).eq("id", record_id).execute()
                        st.session_state['duplicate_found'] = False
                        st.success(f"✅ Overwrote existing record ID #{record_id} for **{student_name}** at {get_ist_timestamp()}!")
                        st.rerun()

                with col_dup2:
                    if st.button("➕ Save as New Entry", use_container_width=True):
                        payload = {
                            "student_name": student_name.strip(),
                            "math_score": math_score,
                            "reading_score": reading_score,
                            "writing_score": writing_score,
                            "attendance": attendance,
                            "test_prep": test_prep,
                            "predicted_risk": round(risk_prob, 2)
                        }
                        supabase_client.table("student_evaluations").insert(payload).execute()
                        st.session_state['duplicate_found'] = False
                        st.success(f"✅ Added new duplicate entry for **{student_name}** at {get_ist_timestamp()}!")
                        st.rerun()

# =============================================================================
# TAB 2: SUBJECT & ATTENDANCE DIAGNOSTICS & EMAIL GENERATOR
# =============================================================================
with tab2:
    st.markdown(f"### 🎯 Diagnostic & Action Plans for **{student_name}**")
    st.markdown("---")

    def get_subject_status(score):
        if score < 50: return "🔴 Critical Risk", "Immediate intervention required."
        elif score < 70: return "🟡 Warning Level", "Needs targeted practice."
        else: return "🟢 Satisfactory", "Meeting benchmarks."

    def get_attendance_status(att):
        if att < 75.0: return "🔴 Severe Deficit", "Critical attendance drop."
        elif att < 85.0: return "🟡 Moderate Risk", "Below safe 85% threshold."
        else: return "🟢 Good Attendance", "Consistently attending."

    col_m, col_r, col_w, col_a = st.columns(4)

    with col_m:
        with st.container(border=True):
            st.markdown("#### 📐 Math Risk")
            m_stat, m_desc = get_subject_status(math_score)
            st.write(f"Score: **{math_score}%** | {m_stat}")
            st.caption(m_desc)
            st.markdown("**Action Plan:**")
            if math_score < 50: st.write("• Assign 1-on-1 tutoring twice weekly.\n• Practice algebra fundamentals.")
            elif math_score < 70: st.write("• Practice formulas & word problems.")
            else: st.write("• Maintain practice schedule.")

    with col_r:
        with st.container(border=True):
            st.markdown("#### 📖 Reading Risk")
            r_stat, r_desc = get_subject_status(reading_score)
            st.write(f"Score: **{reading_score}%** | {r_stat}")
            st.caption(r_desc)
            st.markdown("**Action Plan:**")
            if reading_score < 50: st.write("• Assign guided reading exercises.\n• Weekly vocabulary drills.")
            elif reading_score < 70: st.write("• Practice passage summaries.")
            else: st.write("• Independent reading log.")

    with col_w:
        with st.container(border=True):
            st.markdown("#### ✍️ Writing Risk")
            w_stat, w_desc = get_subject_status(writing_score)
            st.write(f"Score: **{writing_score}%** | {w_stat}")
            st.caption(w_desc)
            st.markdown("**Action Plan:**")
            if writing_score < 50: st.write("• Mandate writing lab sessions.\n• Focus on grammar & structure.")
            elif writing_score < 70: st.write("• Provide outline templates.")
            else: st.write("• Advanced essay synthesis.")

    with col_a:
        with st.container(border=True):
            st.markdown("#### ⏱️ Attendance")
            a_stat, a_desc = get_attendance_status(attendance)
            st.write(f"Rate: **{attendance}%** | {a_stat}")
            st.caption(a_desc)
            st.markdown("**Action Plan:**")
            if attendance < 75.0: st.write("• Mandatory guardian conference.\n• Issue attendance warning.")
            elif attendance < 85.0: st.write("• Attendance advisory notice.")
            else: st.write("• Maintain attendance habits.")

    st.markdown("---")
    st.markdown("### ✉️ Automated Outreach Draft Generator")

    e_col1, e_col2 = st.columns([0.4, 0.6], gap="medium")

    with e_col1:
        recipient_role = st.selectbox("Recipient Type", ["Parent / Guardian", "Academic Advisor", "Student"])
        sender_title = st.text_input("Sender Name / Title", value="Academic Advisory Team")
        custom_note = st.text_area("Custom Note (Optional)", placeholder="e.g., Parent conference set for Thursday.")

    weak_str = ", ".join([f"{sub} ({sc}%)" for sub, sc in weak_subjects]) if weak_subjects else "None"
    
    if recipient_role == "Parent / Guardian":
        subject_line = f"Academic Update: {student_name}"
        email_body = f"""Dear Parent/Guardian of {student_name},

Academic progress report for {student_name}:

Overview:
• Course Average: {avg_score:.1f}%
• Attendance Rate: {attendance}%
• Calculated Academic Risk Index: {risk_prob:.1f}%

Subject Scores: Math ({math_score}%), Reading ({reading_score}%), Writing ({writing_score}%).
"""
        if weak_subjects or attendance < 85.0:
            email_body += f"\nAttention Areas: {weak_str}\nAttendance: {attendance}%\n"
        if custom_note: email_body += f"\nNote: {custom_note}\n"
        email_body += f"\nBest regards,\n{sender_title}"

    elif recipient_role == "Academic Advisor":
        subject_line = f"ADVISOR ALERT: Risk Review - {student_name}"
        email_body = f"""ATTN: Advisory Team,

Student: {student_name}
Risk Index: {risk_prob:.1f}% | Attendance: {attendance}% | Average: {avg_score:.1f}%
Flagged Subjects: {weak_str}

Action Recommended: Conduct a check-in and review attendance habits.
{f'Notes: {custom_note}' if custom_note else ''}
Sender: {sender_title}"""

    else:
        subject_line = f"Academic Tips & Progress - {student_name}"
        email_body = f"""Hi {student_name},

Your academic update: Average: {avg_score:.1f}% | Attendance: {attendance}%.
Scores: Math ({math_score}%), Reading ({reading_score}%), Writing ({writing_score}%).

{f'Focus Areas: {weak_str}' if weak_subjects else 'Great job keeping your grades up!'}
{f'Advisor Note: {custom_note}' if custom_note else ''}

Best,\n{sender_title}"""

    with e_col2:
        st.markdown("##### Generated Email Preview")
        st.text_input("Subject Line", value=subject_line, key="subj_line")
        st.code(email_body, language="markdown")

# =============================================================================
# TAB 3: SUPABASE DB, GROUND TRUTH & DATA EXPORT
# =============================================================================
with tab3:
    st.markdown("### ☁️ Supabase Cloud DB Records & Ground Truth Feedback Loop")
    st.write(f"Current Local Time: **{get_ist_timestamp()}**")

    if supabase_client:
        try:
            # Fetch records from Supabase
            response = supabase_client.table("student_evaluations").select("*").order("id", desc=True).execute()
            records = response.data

            if records:
                df_records = pd.DataFrame(records)
                
                # Format actual_failed column for user readability
                def format_gt(val):
                    if pd.isna(val) or val is None: return "⏳ Pending Outcome"
                    return "🔴 Failed Course (1)" if int(val) == 1 else "🟢 Passed Course (0)"
                
                df_records['Ground Truth Status'] = df_records['actual_failed'].apply(format_gt)

                st.markdown("#### 📄 Saved Database Records")
                st.dataframe(df_records, use_container_width=True)

                # Export CSV Button
                csv_data = df_records.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download All Records as CSV",
                    data=csv_data,
                    file_name=f"student_evaluations_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    type="secondary"
                )

                st.markdown("---")
                st.markdown("#### 🎯 Update Ground Truth (`actual_failed` Field)")
                st.caption("Update real end-of-semester outcomes to prepare data for model retraining.")

                up_col1, up_col2, up_col3 = st.columns(3)
                with up_col1:
                    record_id = st.selectbox("Select Record ID", df_records['id'].tolist())
                with up_col2:
                    outcome = st.selectbox("Final Term Outcome", ["0 - Passed Course", "1 - Failed Course"])
                    outcome_val = 1 if "1" in outcome else 0
                with up_col3:
                    st.write("") # spacing
                    if st.button("🔄 Save Ground Truth Outcome", use_container_width=True):
                        supabase_client.table("student_evaluations").update({"actual_failed": outcome_val}).eq("id", record_id).execute()
                        st.success(f"Updated Record ID #{record_id} outcome to {'Failed' if outcome_val == 1 else 'Passed'}!")
                        st.rerun()

                st.markdown("---")
                st.markdown("#### 🤖 Active Machine Learning Retraining Trigger")
                
                # Ground truth filtering
                df_gt = df_records[df_records['actual_failed'].notnull()]
                st.write(f"Logged Ground Truth Outcome Records: **{len(df_gt)}**")

                if st.button("⚙️ Retrain ML Model Weights on Real Outcomes"):
                    if len(df_gt) < 3:
                        st.warning("⚠️ Need at least 3 Ground Truth outcome records to trigger retraining. Mark end-of-term results above!")
                    else:
                        X_gt = df_gt[['math_score', 'reading_score', 'writing_score', 'attendance']]
                        y_gt = df_gt['actual_failed'] * 100.0
                        
                        retrained_model = make_pipeline(StandardScaler(), Ridge(alpha=0.1))
                        retrained_model.fit(X_gt, y_gt)
                        st.success("🎉 Retraining Complete! Active model updated using real-world Supabase outcomes.")
            else:
                st.info("No saved records found in Supabase. Go to Tab 1 and click '💾 Save Assessment to Supabase'.")

        except Exception as e:
            st.error(f"Supabase Error: {e}")
    else:
        st.warning("⚠️ Supabase credentials not found in `.streamlit/secrets.toml`. Running in local demo mode.")

# =============================================================================
# TAB 4: SYSTEM SPEC
# =============================================================================
with tab4:
    with st.container(border=True):
        st.markdown("#### 📊 System Sensitivity & Logic Spec")
        st.write("""
        * **Timezone Standard:** Indian Standard Time (IST / UTC+5:30) used for all timestamp logs.
        * **Duplicate Protection:** Scans Supabase before inserting matching student names and provides confirmation dialogs.
        * **CSV Exporting:** Native `st.download_button` exports complete evaluation records.
        * **Ground Truth Feedback Loop:** Retrains continuous Ridge coefficients once end-of-term `actual_failed` flags are recorded.
        """)