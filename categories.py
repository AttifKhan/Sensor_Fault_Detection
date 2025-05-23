class DocumentCategories:
    """Define document categories with highly specific prompts for CLIP"""
    
    CATEGORIES = {
    "driving_license": [
        "Driving Licence"
    ],
    "passport": [
      "Passport"
    ],
    "bank_details": [
        "Bank Details", "Passbook", "Account", "Cheque",
        "Checkbook with routing number and account details"
    ],
    "employment_verification": [
        "Employeement Eligibility Verification form",
        "letter on company letterhead stating employee’s job title",
        "Form I-9 section C with employer certification date",
    ],
    "social_security": [
        "document listing a nine-digit SSN in microprint",
        "laminated card bearing issue date and signature line",
        "SSA"
    ],
   
    "tax_documents": [
        "tax",
        "W-2",
        "Form 1040",
        "Federal or state tax filing paperwork"
    ],
    
    "education_certificate": [
        "Diploma showing institution name, embossed seal, degree awarded, and conferral date",
        "Official transcript with institution letterhead, course–credit table, GPA calculation, and registrar’s signature",
        "Certificate bearing institution seal/stamp, serial number, degree title, and date of conferral",
        "Degree parchment with Board of Education header, authorized signature block, and document ID",
        "Academic marks sheet with institution logo, student details, subject-wise marks table, and registrar sign-off"
    ],
    "others": [
        "Unrelated to driving license, passport, bank details, employment verification, social_security, tax documents, or education certificate",
        "Any random photo if it’s not a document (e.g., pets, landscapes, objects, characters)",
        "Movie, cartoon, or video game character images",
        "Generic correspondence unrelated to official verification",
        "Miscellaneous paper not used for licensing or certification (e.g., receipts, flyers)",
        "Personal note without any government or financial seals",
        "Handwritten memo with no official identifiers",
        "Blank or decorative stationery"
    ]
    }