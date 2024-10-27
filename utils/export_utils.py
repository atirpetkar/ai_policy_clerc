import json
import pandas as pd
import streamlit as st
from datetime import datetime
import io

def export_chat_history(messages, format='json'):
    """Export chat history in specified format"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if format == 'json':
            # Convert messages to JSON format
            export_data = json.dumps({
                'timestamp': timestamp,
                'messages': messages
            }, indent=2)
            
            # Create download button
            st.download_button(
                label="📥 Download Chat History (JSON)",
                data=export_data,
                file_name=f"chat_history_{timestamp}.json",
                mime="application/json"
            )
        elif format == 'csv':
            # Convert messages to DataFrame
            df = pd.DataFrame(messages)
            csv_data = df.to_csv(index=False)
            
            # Create download button
            st.download_button(
                label="📥 Download Chat History (CSV)",
                data=csv_data,
                file_name=f"chat_history_{timestamp}.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error exporting chat history: {str(e)}")

def export_analysis_results(data, analysis_type, format='csv'):
    """Export analysis results in specified format"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if isinstance(data, pd.DataFrame):
            if format == 'csv':
                csv_data = data.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download {analysis_type} (CSV)",
                    data=csv_data,
                    file_name=f"{analysis_type.lower().replace(' ', '_')}_{timestamp}.csv",
                    mime="text/csv"
                )
            elif format == 'json':
                json_data = data.to_json(orient='records', indent=2)
                st.download_button(
                    label=f"📥 Download {analysis_type} (JSON)",
                    data=json_data,
                    file_name=f"{analysis_type.lower().replace(' ', '_')}_{timestamp}.json",
                    mime="application/json"
                )
        elif isinstance(data, (list, dict)):
            json_data = json.dumps(data, indent=2)
            st.download_button(
                label=f"📥 Download {analysis_type} (JSON)",
                data=json_data,
                file_name=f"{analysis_type.lower().replace(' ', '_')}_{timestamp}.json",
                mime="application/json"
            )
    except Exception as e:
        st.error(f"Error exporting analysis results: {str(e)}")
