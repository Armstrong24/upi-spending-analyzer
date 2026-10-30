"""Current limitations and the future-scope roadmap (shown in both the web and desktop apps)."""

LIMITATIONS = [
    ("Synthetic training data",
     "The model learned from 3,000 generated descriptions, so it scores 99.8% on familiar merchants "
     "but about 62% on shops it has never seen."),
    ("Manual statement download",
     "The user has to export a CSV/PDF from PhonePe, GPay, Paytm or the bank and open it in the app."),
    ("Fixed PDF layouts",
     "PDF formats differ between apps and banks; we read the PhonePe-style line format. Scanned PDFs are not read."),
    ("Hand-tuned fraud thresholds",
     "z-score 2.5, late-night hours and scam keywords are rules we chose; there is no labelled real fraud data."),
    ("Past data only",
     "It explains where money went, but does not yet predict next month or alert you while you spend."),
]

# phase, feature, what it adds, how (technology), FLB concept it connects to
ROADMAP = [
    ("Phase 1: Next 3 months", "Learn from user corrections",
     "User fixes a wrong category; the model retrains and remembers it for that merchant.",
     "Active learning, per-user merchant memory", "Personal financial planning"),
    ("Phase 1: Next 3 months", "More statement formats + OCR",
     "Read GPay, Paytm, SBI, HDFC, ICICI PDFs and scanned or photographed statements.",
     "pdfplumber templates, Tesseract OCR", "Digital payments (Unit 5)"),
    ("Phase 1: Next 3 months", "Budget goals and alerts",
     "Set a goal (e.g. save ₹10,000 for a laptop) and get warned when a category crosses its limit.",
     "Rule engine, desktop notifications", "Budgeting, savings habits"),
    ("Phase 1: Next 3 months", "Hindi and Marathi support",
     "Understand regional merchant names and show the app in Indian languages.",
     "Multilingual tokenisation, Qt translations", "Financial inclusion"),
    ("Phase 2: 6-12 months", "Automatic statements via Account Aggregator",
     "Fetch transactions directly from the bank with the user's consent, no manual download.",
     "RBI Account Aggregator framework (Sahamati), FIU licence", "Fintech, Open banking (Unit 5)"),
    ("Phase 2: 6-12 months", "Spending forecast",
     "Predict next month's spend per category and warn early about overspending.",
     "Time series: ARIMA / Prophet / LSTM", "Financial planning, cash-flow management"),
    ("Phase 2: 6-12 months", "Smarter categoriser",
     "Lift unseen-merchant accuracy from 62% using language models trained on real labelled data.",
     "IndicBERT / DistilBERT fine-tuning", "AI in fintech (Unit 5)"),
    ("Phase 2: 6-12 months", "Private offline AI coach",
     "Personalised advice from a small LLM running on the laptop, no API key or internet.",
     "Local LLM (e.g. Phi / Llama via Ollama)", "Fintech, data privacy"),
    ("Phase 3: Long term", "Real-time fraud shield",
     "Score each payment as it happens, check payee VPAs against reported scam lists, explain why.",
     "Graph analysis of payee networks, SHAP explanations, 1930 helpline data", "Banking frauds (Unit 3)"),
    ("Phase 3: Long term", "Android app with SMS tracking",
     "Track spending automatically from bank SMS alerts (with permission) on the phone.",
     "Kotlin / Flutter, on-device ML (TF Lite)", "Digital payments, mobile banking"),
    ("Phase 3: Long term", "Invest the savings",
     "Turn monthly surplus into goal-based SIP / RD suggestions and credit-score (CIBIL) health tips.",
     "SIP/RD calculators, compounding projections", "Savings & investment, credit"),
    ("Phase 3: Long term", "Security and compliance",
     "Encrypted local storage, consent logs, code-signed installer with auto-update.",
     "AES encryption, DPDP Act 2023 compliance, code signing", "RBI customer protection, data privacy"),
]

IMPACT = ("Goal: grow from a student budgeting tool into a privacy-first personal finance assistant that helps "
          "first-time UPI users in India spend wisely, save regularly and avoid fraud.")
