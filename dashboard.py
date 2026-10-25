import streamlit as st
import pandas as pd
import os
import sys
import subprocess

st.set_page_config(
    page_title="Multimodal Attendance Agent",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Multimodal Attendance & Engagement Agent")
st.write("Start attendance using your camera and view session analytics.")

# Start Attendance Button
st.subheader("📷 Camera Attendance")

if st.button("▶️ Start Attendance", type="primary"):
    if os.path.exists("multimodal_agent.py"):
        with st.spinner("Attendance session is running. Please wait..."):
            try:
                result = subprocess.run(
                    [sys.executable, "multimodal_agent.py"],
                    capture_output=True,
                    text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )

                if result.returncode == 0:
                    st.success("Attendance session completed!")
                else:
                    st.error("The attendance program encountered an error.")

                if result.stdout:
                    st.subheader("📋 Attendance Session Report")
                    st.code(result.stdout)

                if result.stderr:
                    with st.expander("View error details"):
                        st.code(result.stderr)

            except Exception as e:
                st.error(f"Could not start attendance: {e}")
    else:
        st.error("multimodal_agent.py was not found in the project folder.")

# Load Attendance Data
st.divider()
st.subheader("📋 Attendance Records")

if os.path.exists("attendance.csv"):
    attendance = pd.read_csv("attendance.csv")
    st.dataframe(attendance, use_container_width=True)
    st.metric("Total Attendance Records", len(attendance))
else:
    st.warning("attendance.csv not found. Run an attendance session first.")

# Load Session Analytics
st.divider()
st.subheader("📊 Session Analytics")

if os.path.exists("session_analytics.csv"):
    analytics = pd.read_csv("session_analytics.csv")
    st.dataframe(analytics, use_container_width=True)

    if not analytics.empty:
        latest = analytics.iloc[-1]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Students Recognized",
            int(latest["Students_Recognized"])
        )

        col2.metric(
            "Audio Activity",
            f'{latest["Audio_Activity_Percentage"]}%'
        )

        col3.metric(
            "Experimental Engagement Score",
            f'{latest["Engagement_Score"]}/100'
        )

        st.subheader("📈 Engagement Score by Session")

        chart_data = analytics[["Engagement_Score"]].copy()
        chart_data.index = range(1, len(chart_data) + 1)
        st.line_chart(chart_data)

        st.caption(
            "Engagement score is experimental. Audio activity "
            "does not prove student attention."
        )
else:
    st.info(
        "session_analytics.csv not found. "
        "Complete an attendance session to generate analytics."
    )

st.divider()
st.caption("Multimodal Attendance Agent | Student Project Prototype")