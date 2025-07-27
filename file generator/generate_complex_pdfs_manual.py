import os
import random
import json
from datetime import date, timedelta, time

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch

# --- CONFIGURATION ---
NUM_DOCS = 50
output_pdf_dir = "generated_invitations_pdf"
output_json_dir = "generated_invitations_json"

# --- SETUP DIRECTORIES ---
os.makedirs(output_pdf_dir, exist_ok=True)
os.makedirs(output_json_dir, exist_ok=True)


# --- DIVERSE DUMMY DATA POOLS ---
FIRST_NAMES = ["Liam", "Olivia", "Noah", "Emma", "Oliver", "Ava", "Elijah", "Charlotte", "William", "Sophia"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez"]
PARTY_FOR_REASON = ["Birthday", "Team Victory", "Graduation", "End of School Year", "Welcome Home"]

VENUES = [
    ("SKY HIGH TRAMPOLINE ARENA", "123 Main Street", "Phoenix, AZ 85001"),
    ("GRAVITY ZONE PARK", "456 Oak Avenue", "Austin, TX 78701"),
    ("JUMP FUSION", "789 Pine Lane", "Denver, CO 80202"),
    ("AERIAL FUN CENTER", "101 Maple Drive", "Miami, FL 33101"),
    ("BOUNCE WORLD", "212 Birch Road", "Seattle, WA 98101")
]

RSVP_CONTACTS = [("Jennifer", "555-0101"), ("Mike", "555-0102"), ("Sarah", "555-0103"), ("David", "555-0104")]


# --- STYLESHEET SETUP ---
def get_custom_styles():
    """Returns a stylesheet with custom paragraph styles for the invitation."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="VenueTitle", fontSize=24, fontName="Helvetica-Bold", alignment=1, spaceAfter=6, textColor=colors.darkblue))
    styles.add(ParagraphStyle(name="SubTitle", fontSize=14, fontName="Helvetica-Bold", alignment=1, spaceAfter=20, textColor=colors.darkblue))
    styles.add(ParagraphStyle(name="InviteHeader", fontSize=28, fontName="Helvetica-Bold", alignment=1, leading=34))
    styles.add(ParagraphStyle(name="PartyText", fontSize=48, fontName="Helvetica-Bold", alignment=1, spaceAfter=24, textColor=colors.red))
    styles.add(ParagraphStyle(name="InfoLabel", fontSize=12, fontName="Helvetica-Bold", alignment=0))
    styles.add(ParagraphStyle(name="InfoData", fontSize=12, fontName="Helvetica", alignment=0))
    styles.add(ParagraphStyle(name="FooterText", fontSize=9, fontName="Helvetica", alignment=1, leading=12, textColor=colors.dimgrey))
    return styles

STYLES = get_custom_styles()


# --- DATA GENERATION FUNCTIONS ---
def get_random_future_date():
    """Generates a random date within the next 90 days."""
    start_date = date.today()
    random_days = random.randint(14, 90)
    return (start_date + timedelta(days=random_days)).strftime("%A, %B %d, %Y")

def get_random_time():
    """Generates a random time between 10 AM and 7 PM."""
    rand_hour = random.randint(10, 19)
    rand_minute = random.choice([0, 15, 30, 45])
    return time(hour=rand_hour, minute=rand_minute).strftime("%I:%M %p")

def generate_invitation_data():
    """Generates a dictionary of random data for a single invitation."""
    venue_name, venue_street, venue_city = random.choice(VENUES)
    rsvp_name, rsvp_phone = random.choice(RSVP_CONTACTS)
    
    data = {
        "venue_name": venue_name,
        "for_person": f"{random.choice(FIRST_NAMES)}'s {random.choice(PARTY_FOR_REASON)} Party",
        "date": get_random_future_date(),
        "time": get_random_time(),
        "address_l1": venue_street,
        "address_l2": venue_city,
        "rsvp": f"by {date.today() + timedelta(days=7):%B %d} to {rsvp_name} at {rsvp_phone}"
    }
    return data


# --- DOCUMENT AND JSON GENERATION ---
def create_invitation_pdf(filename, data):
    """Creates a single party invitation PDF with the provided unique data."""
    doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)
    story = []

    story.append(Paragraph(data['venue_name'], STYLES["VenueTitle"]))
    story.append(Paragraph("TRAMPOLINE PARK", STYLES["SubTitle"]))
    story.append(Paragraph("YOU'RE INVITED<br/>TO A", STYLES["InviteHeader"]))
    story.append(Paragraph("PARTY", STYLES["PartyText"]))
    
    # Information Table
    info_data = [
        [Paragraph("FOR:", STYLES['InfoLabel']), Paragraph(data['for_person'], STYLES['InfoData'])],
        [Paragraph("DATE:", STYLES['InfoLabel']), Paragraph(data['date'], STYLES['InfoData'])],
        [Paragraph("TIME:", STYLES['InfoLabel']), Paragraph(data['time'], STYLES['InfoData'])],
        [Paragraph("ADDRESS:", STYLES['InfoLabel']), Paragraph(f"{data['address_l1']}<br/>{data['address_l2']}", STYLES['InfoData'])],
        [Paragraph("RSVP:", STYLES['InfoLabel']), Paragraph(data['rsvp'], STYLES['InfoData'])],
    ]
    
    info_table = Table(info_data, colWidths=[0.8*inch, 4.5*inch])
    info_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5*inch))

    # Footer
    story.append(Paragraph("CLOSED-TOED SHOES ARE REQUIRED FOR CLIMBING ACTIVITIES.", STYLES["FooterText"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("PARENTS OR GUARDIANS NOT ATTENDING, PLEASE VISIT OUR WEBSITE TO FILL OUT A WAIVER.", STYLES["FooterText"]))
    story.append(Spacer(1, 24))
    story.append(Paragraph("HOPE TO SEE YOU THERE!", STYLES["VenueTitle"]))

    doc.build(story)

def make_json(filename):
    """Saves the static data to a JSON file in the specified format."""
    json_output_data = {
        "title": "",
        "outline": [
            {
                "level": "H1",
                "text": "HOPE To SEE You THERE! ",
                "page": 0
            }
        ]
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_output_data, f, indent=4, ensure_ascii=False)

# --- MAIN EXECUTION ---
def main():
    """Generates a specified number of unique PDF invitations and their corresponding JSON data files."""
    for i in range(1, NUM_DOCS + 1):
        base_filename = f"Party_Invitation_{i:03d}"
        pdf_filename = os.path.join(output_pdf_dir, f"{base_filename}.pdf")
        json_filename = os.path.join(output_json_dir, f"{base_filename}.json")
        
        print(f"({i}/{NUM_DOCS}) Generating {pdf_filename} and {json_filename}...")
        
        # Generate a unique set of data for the invitation
        invitation_data = generate_invitation_data()
        
        # Create the PDF invitation with the unique data
        create_invitation_pdf(pdf_filename, invitation_data)
        
        # Create the corresponding static JSON file
        make_json(json_filename)

    print(f"\n✅ Done. {NUM_DOCS} PDF invitations and JSON files have been generated.")
    print(f"   PDFs are in: '{output_pdf_dir}'")
    print(f"   JSONs are in: '{output_json_dir}'")

if __name__ == "__main__":
    main()
