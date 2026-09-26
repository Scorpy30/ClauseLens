from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


OUTPUT_PATH = Path(__file__).parent / "sample.pdf"


def create_sample_pdf() -> None:
    pdf = canvas.Canvas(str(OUTPUT_PATH), pagesize=A4)

    # Page 1
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(72, 780, "SAMPLE EMPLOYMENT AGREEMENT")

    pdf.setFont("Helvetica", 11)

    lines = [
        "1. Employment",
        "",
        "The Employee agrees to perform the duties assigned by the Company",
        "and to comply with the policies applicable to their position.",
        "",
        "2. Compensation",
        "",
        "The Employee will receive an annual base salary of INR 900,000,",
        "payable in accordance with the Company's normal payroll schedule.",
        "",
        "3. Confidentiality",
        "",
        "The Employee must keep confidential all non-public business",
        "information obtained during the course of employment.",
    ]

    y = 745

    for line in lines:
        pdf.drawString(72, y, line)
        y -= 18

    pdf.showPage()

    # Page 2
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(72, 780, "4. Termination and Notice")

    pdf.setFont("Helvetica", 11)

    lines = [
        "",
        "Either party may terminate the employment relationship by providing",
        "60 days' written notice to the other party.",
        "",
        "The Company may waive all or part of the notice period at its discretion.",
        "",
        "5. Post-Employment Restrictions",
        "",
        "For six months following termination, the Employee must not use",
        "confidential Company information for the benefit of a competitor.",
    ]

    y = 745

    for line in lines:
        pdf.drawString(72, y, line)
        y -= 18

    pdf.save()


if __name__ == "__main__":
    create_sample_pdf()
    print(f"Created: {OUTPUT_PATH}")
