
import io
import fitz  # PyMuPDF
import pandas as pd
import google.generativeai as genai
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import time
import re
import os
from dotenv import load_dotenv


# --- Load environment variables ---
load_dotenv()

# --- Configuration ---
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE', 'cred.json')
FOLDER_ID = os.getenv('FOLDER_ID')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# --- Google Drive Authentication ---
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
drive_service = build("drive", "v3", credentials=creds)

# --- Helper Functions ---
def get_pdfs_from_folder(folder_id):
    """Get all PDF files from the specified folder in a Shared Drive"""
    pdfs = []
    print("🔍 Searching for PDF files in the shared drive folder...")
    
    # Query to find all PDF files in the folder
    query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    
    try:
        # For shared drives, we need to include supportsAllDrives=True
        results = drive_service.files().list(
            q=query, 
            fields="files(id, name, mimeType, createdTime, modifiedTime)",
            pageSize=1000,
            supportsAllDrives=True,  # Required for shared drives
            includeItemsFromAllDrives=True  # Required for shared drives
        ).execute()
        
        files = results.get('files', [])
        
        for file in files:
            pdfs.append({
                "id": file['id'],
                "name": file['name'],
                "created_time": file.get('createdTime', ''),
                "modified_time": file.get('modifiedTime', '')
            })
            print(f"📄 Found PDF: {file['name']}")
        
        print(f"✅ Found {len(pdfs)} PDF files.")
        return pdfs
        
    except Exception as e:
        print(f"❌ Error accessing shared drive: {e}")
        # Check if it's a permissions issue
        if "insufficientFilePermissions" in str(e):
            print("💡 Make sure your service account has been granted access to the shared drive")
        return []

def download_pdf(file_id):
    """Download a PDF file from Google Drive (including Shared Drive)"""
    try:
        # For shared drives, we need to include supportsAllDrives=True
        request = drive_service.files().get_media(
            fileId=file_id,
            supportsAllDrives=True  # Required for shared drives
        )
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        
        while not done:
            status, done = downloader.next_chunk()
            print(f"⬇️ Download progress: {int(status.progress() * 100)}%")
        
        fh.seek(0)
        return fh.getvalue()
    except Exception as e:
        print(f"❌ Error downloading file {file_id}: {e}")
        return None

def extract_text_from_pdf(pdf_content):
    """Extract text from PDF content"""
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        return ""

def extract_cv_data_with_gemini(cv_text, cv_name):
    """Use Gemini AI to extract CV data"""
    # Truncate text to avoid token limits while preserving important sections
    truncated_text = cv_text[:10000]  # Reduced from 12000 to be safer
    
    prompt = f"""
    Analyze this CV and extract the following information in a structured format. 
    If information is not available, write "Not specified".
    
    CV Content:
    {truncated_text}
    
    Please extract the following information:
    1. Candidate Name
    2. Email Address
    3. Phone Number
    4. Current Company/Organization
    5. Current Position/Title
    6. Total Years of Experience (as a number)
    7. Work History separte coloums as carrer Journey (list companies, positions, and durations in chronological order)
    8. Education History(degrees, institutions, years)
    9. Key Skills/Technologies (as a comma-separated list)
    10. Certifications that matches the work history (if any)
    11. Notice Period (if mentioned)
    12. Current Location
    13. Preferred Location(s)
    14. Current Salary (if mentioned)
    15. Expected Salary (if mentioned)
    16. Has the candidate switched jobs multiple times? (Yes/No based on work history)
    17. Overall Summary of key accomplishment during the carrer journey. Highlight the tools used and projects roles (brief 3-4 sentence professional summary)
    
    Format your response exactly as follows without any additional text or markdown:
    Name: [extracted name]
    Email: [extracted email]
    Phone: [extracted phone number]
    Current Company: [extracted current company]
    Current Position: [extracted current position]
    Total Experience: [number] years
    Work History: as separte cooumns [company1 - position1 (duration); company2 - position2 (duration); ...] 
    Experience 1: [company1 - position1 (duration)]
    Experience 2: [company2 - position2 (duration)] 
    experience 3: [company3 - position3 (duration)]
    Experience 4: [company4 - position4 (duration)]
    experience 5: [company5 - position5 (duration)]
    Experience6: [company6 - position6 (duration)]
    Experience 7: [company7 - position7 (duration)]
    Experience8 : [company8 - position8 (duration)]
    Experience9 : [company9 - position9 (duration)]
    Experience10: [company10 - position10 (duration)]  
    Education: as milestones with year if mentioned  [degree1 - institution1 (year); degree2 - institution2 (year); ...]
    Key Skills: [skill1, skill2, skill3, ...]
    Certifications: [certification1, certification2, ...]
    Notice Period: [notice period if mentioned]
    Current Location: [current location]
    Preferred Location: [preferred location(s)]
    Current Salary: [current salary if mentioned]
    Expected Salary: [expected salary if mentioned]
    Job Switching: [Yes/No]
    Summary: [brief summary]
    """
    
    try:
        response = model.generate_content(prompt)
        return parse_gemini_response(response.text, cv_name)
    except Exception as e:
        print(f"❌ Error with Gemini API for {cv_name}: {e}")
        return create_error_response(cv_name, str(e))

def parse_gemini_response(response_text, cv_name):
    """Parse the response from Gemini into a structured dictionary"""
    lines = response_text.split('\n')
    data = {
        "CV File Name": cv_name,
        "Name": "Not specified",
        "Email": "Not specified",
        "Phone": "Not specified",
        "Current Company": "Not specified",
        "Current Position": "Not specified",
        "Total Experience": "Not specified",
        "Work History": "Not specified",
        "Experience 1": "Not specified",
        "Experience 2": "Not specified",    
        "Experience 3": "Not specified",
        "Experience 4": "Not specified",
        "Experience 5": "Not specified",
        "Experience 6": "Not specified",
        "Experience 7": "Not specified",
        "Experience 8": "Not specified",
        "Experience 9": "Not specified",
        "Experience 10": "Not specified",
        "Education": "Not specified",
        "Key Skills": "Not specified",
        "Certifications": "Not specified",
        "Notice Period": "Not specified",
        "Current Location": "Not specified",
        "Preferred Location": "Not specified",
        "Current Salary": "Not specified",
        "Expected Salary": "Not specified",
        "Job Switching": "Not specified",
        "Summary": "Not specified",
        "Processing Error": None
    }
    
    for line in lines:
        line = line.strip()
        if line.startswith("Name:"):
            data["Name"] = line.replace("Name:", "").strip()
        elif line.startswith("Email:"):
            data["Email"] = line.replace("Email:", "").strip()
        elif line.startswith("Phone:"):
            data["Phone"] = line.replace("Phone:", "").strip()
        elif line.startswith("Current Company:"):
            data["Current Company"] = line.replace("Current Company:", "").strip()
        elif line.startswith("Current Position:"):
            data["Current Position"] = line.replace("Current Position:", "").strip()
        elif line.startswith("Total Experience:"):
            data["Total Experience"] = line.replace("Total Experience:", "").strip()
        elif line.startswith("Work History:"):
            data["Work History"] = line.replace("Work History:", "").strip()
        elif line.startswith("Experience 1:"):
            data["Experience 1"] = line.replace("Experience 1:", "").strip()
        elif line.startswith("Experience 2:"):
            data["Experience 2"] = line.replace("Experience 2:", "").strip()
        elif line.startswith("Experience 3:"):
            data["Experience 3"] = line.replace("Experience 3:", "").strip()
        elif line.startswith("Experience 4:"):
            data["Experience 4"] = line.replace("Experience 4:", "").strip()
        elif line.startswith("Experience 5:"):
            data["Experience 5"] = line.replace("Experience 5:", "").strip()
        elif line.startswith("Experience 6:"):
            data["Experience 6"] = line.replace("Experience 6:", "").strip()
        elif line.startswith("Experience 7:"):
            data["Experience 7"] = line.replace("Experience 7:", "").strip()
        elif line.startswith("Experience 8:"):
            data["Experience 8"] = line.replace("Experience 8:", "").strip()
        elif line.startswith("Experience 9:"):
            data["Experience 9"] = line.replace("Experience 9:", "").strip()
        elif line.startswith("Experience 10:"):
            data["Experience 10"] = line.replace("Experience 10:", "").strip()
        elif line.startswith("Education:"):
            data["Education"] = line.replace("Education:", "").strip()
        elif line.startswith("Key Skills:"):
            data["Key Skills"] = line.replace("Key Skills:", "").strip()
        elif line.startswith("Certifications:"):
            data["Certifications"] = line.replace("Certifications:", "").strip()
        elif line.startswith("Notice Period:"):
            data["Notice Period"] = line.replace("Notice Period:", "").strip()
        elif line.startswith("Current Location:"):
            data["Current Location"] = line.replace("Current Location:", "").strip()
        elif line.startswith("Preferred Location:"):
            data["Preferred Location"] = line.replace("Preferred Location:", "").strip()
        elif line.startswith("Current Salary:"):
            data["Current Salary"] = line.replace("Current Salary:", "").strip()
        elif line.startswith("Expected Salary:"):
            data["Expected Salary"] = line.replace("Expected Salary:", "").strip()
        elif line.startswith("Job Switching:"):
            data["Job Switching"] = line.replace("Job Switching:", "").strip()
        elif line.startswith("Summary:"):
            data["Summary"] = line.replace("Summary:", "").strip()
    
    return data

def create_error_response(cv_name, error_msg):
    """Create a response dictionary for errored CV processing"""
    return {
        "CV File Name": cv_name,
        "Name": "Error in processing",
        "Email": "Error in processing",
        "Phone": "Error in processing",
        "Current Company": "Error in processing",
        "Current Position": "Error in processing",
        "Total Experience": "Error in processing",
        "Work History": "Error in processing",
        "Experience 1": "Error in processing",
        "Experience 2": "Error in processing",
        "Experience 3": "Error in processing",
        "Experience 4": "Error in processing",
        "Experience 5": "Error in processing",
        "Experience 6": "Error in processing",
        "Experience 7": "Error in processing",
        "Experience 8": "Error in processing",
        "Experience 9": "Error in processing",
        "Experience 10": "Error in processing",
        "Education": "Error in processing",
        "Key Skills": "Error in processing",
        "Certifications": "Error in processing",
        "Notice Period": "Error in processing",
        "Current Location": "Error in processing",
        "Preferred Location": "Error in processing",
        "Current Salary": "Error in processing",
        "Expected Salary": "Error in processing",
        "Job Switching": "Error in processing",
        "Summary": "Error in processing",
        "Processing Error": error_msg
    }

def process_cvs_to_dataframe(pdf_list):
    """Process all CVs and return a DataFrame with extracted data"""
    all_cv_data = []
    
    for idx, pdf in enumerate(pdf_list):
        print(f"\n🔍 [{idx+1}/{len(pdf_list)}] Processing: {pdf['name']}")
        
        try:
            # Download PDF
            print("⬇️ Downloading PDF...")
            pdf_content = download_pdf(pdf['id'])
            
            if pdf_content is None:
                print(f"❌ Failed to download {pdf['name']}")
                all_cv_data.append(create_error_response(pdf['name'], "Failed to download PDF"))
                continue
            
            # Extract text
            print("📝 Extracting text from PDF...")
            text = extract_text_from_pdf(pdf_content)
            
            if not text.strip():
                print(f"⚠️ No text extracted from {pdf['name']}")
                all_cv_data.append(create_error_response(pdf['name'], "No text could be extracted from PDF"))
                continue
                
            # Extract data with Gemini
            print("🤖 Analyzing CV with Gemini AI...")
            cv_data = extract_cv_data_with_gemini(text, pdf['name'])
            all_cv_data.append(cv_data)
            
            print(f"✅ Successfully processed: {pdf['name']}")
            
            # Add delay to avoid rate limiting
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Error processing '{pdf['name']}': {e}")
            all_cv_data.append(create_error_response(pdf['name'], str(e)))
    
    return pd.DataFrame(all_cv_data)

# --- Main Execution ---
if __name__ == "__main__":
    print("🚀 Starting CV Data Extraction Process")
    print("======================================")
    
    # Get all PDFs from the folder
    pdfs = get_pdfs_from_folder(FOLDER_ID)
    
    if not pdfs:
        print("❌ No PDFs found in the specified folder.")
        print("💡 Make sure:")
        print("   1. Your service account has access to the shared drive")
        print("   2. The folder ID is correct")
        print("   3. The shared drive contains PDF files")
    else:
        # Process CVs and extract data
        cv_df = process_cvs_to_dataframe(pdfs)
        
        # Save to Excel
        output_file = "cv_extracted_data.xlsx"
        cv_df.to_excel(output_file, index=False)
        
        print(f"\n🎉 CV data extraction complete!")
        print("======================================")
        print(f"📊 Results saved to: {output_file}")
        print(f"📄 Processed {len(cv_df)} CVs")
        
        # Show summary
        success_count = len(cv_df[cv_df['Processing Error'].isna() | (cv_df['Processing Error'] == '')])
        error_count = len(cv_df) - success_count
        
        print(f"✅ Successful: {success_count}")
        print(f"❌ Errors: {error_count}")