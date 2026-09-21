from fpdf import FPDF
from datetime import datetime
import pandas as pd
import os

class PDFReport(FPDF):
    def header(self):
        # Logo placeholder (could add if available)
        self.set_font('helvetica', 'B', 15)
        self.set_text_color(0, 51, 102)  # Dark Blue
        self.cell(0, 10, 'HGG Predictive Maintenance Report', border=False, ln=True, align='C')
        self.set_font('helvetica', 'I', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', border=False, ln=True, align='C')
        # Line width for landscape is 297 - 20 = 277
        self.line(10, 30, 287, 30)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_3_month_pdf(schedule_df: pd.DataFrame, mtbf: float, next_pred_days: float, most_common_fault: str) -> str:
    """
    Generates a PDF report for the next 3 months (90 days) of predicted breakdowns.
    Saves it to a temporary location and returns the file path.
    """
    pdf = PDFReport(orientation='L')
    pdf.add_page()

    # Section 1: Executive Summary
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, 'Executive Summary', ln=True)
    
    pdf.set_font('helvetica', '', 11)
    pdf.set_text_color(0, 0, 0)
    summary_text = (
        f"This report outlines the forecasted maintenance schedule for the HGG CNC Profile Coping Machine "
        f"over the next 3 months (90 days). The predictions are generated using a Weibull survival model "
        f"combined with a Markov Chain transition matrix for fault categorization based on historical data."
    )
    pdf.multi_cell(0, 7, summary_text)
    pdf.ln(5)

    # Metrics
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(50, 7, 'Historical MTBF:')
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 7, f'{mtbf:.1f} days', ln=True)
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(50, 7, 'Next Predicted Fault In:')
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 7, f'{next_pred_days:.0f} days', ln=True)
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(50, 7, 'Primary Risk Area:')
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 7, f'{most_common_fault}', ln=True)
    pdf.ln(10)

    # Section 2: 3-Month Forecast Table
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, '3-Month Predicted Breakdown Schedule', ln=True)
    pdf.ln(3)

    # Table Header
    pdf.set_fill_color(0, 51, 102)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('helvetica', 'B', 10)
    
    # Landscape width is 277mm usable
    col_widths = [15, 40, 30, 50, 142]
    headers = ['#', 'Predicted Date', 'Days Away', 'Most Likely Cause', 'Top 3 Probabilities']
    
    for i in range(len(headers)):
        pdf.cell(col_widths[i], 10, headers[i], border=1, align='C', fill=True)
    pdf.ln()

    # Table Body
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('helvetica', '', 9)
    
    # Filter to next 90 days
    schedule_90 = schedule_df[schedule_df['days_from_now'] <= 90]
    
    if schedule_90.empty:
        pdf.cell(0, 10, 'No breakdowns predicted in the next 90 days.', border=1, align='C')
    else:
        for idx, row in schedule_90.iterrows():
            # Format row data
            num = str(row['event_number'])
            date_str = row['predicted_date'].strftime('%Y-%m-%d')
            days_str = f"{row['days_from_now']:.0f}"
            cause = str(row['predicted_category'])
            
            # Format top 3 (replace | with comma for space)
            top3 = str(row['top_3_categories']).replace(' | ', ', ')
            # No truncation needed in landscape mode
            
            # Print row cells
            # Using multi_cell for the last column to handle wrapping if needed, but cell is easier for simple tables
            pdf.cell(col_widths[0], 8, num, border=1, align='C')
            pdf.cell(col_widths[1], 8, date_str, border=1, align='C')
            pdf.cell(col_widths[2], 8, days_str, border=1, align='C')
            pdf.cell(col_widths[3], 8, cause, border=1, align='C')
            pdf.cell(col_widths[4], 8, top3, border=1, align='L')
            pdf.ln()

    pdf.ln(10)
    
    # Section 3: Recommendations
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, 'Actionable Recommendations', ln=True)
    
    pdf.set_font('helvetica', '', 10)
    pdf.set_text_color(0, 0, 0)
    recs = [
        "- Review spare parts inventory for the 'Primary Risk Area' listed above.",
        "- Schedule preventive maintenance within the 'Days Away' window to avoid unplanned downtime.",
        "- These dates are statistical projections. Ensure routine inspections continue."
    ]
    for rec in recs:
        pdf.cell(0, 8, rec, ln=True)

    # Save PDF
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'HGG_3Month_Forecast.pdf')
    pdf.output(output_path)
    return output_path
