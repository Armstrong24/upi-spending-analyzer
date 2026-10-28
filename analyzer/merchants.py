"""Merchant lists, keyword rules and FLB budget groups.

Everything the app "knows" about Indian merchants lives in this one file,
so it is easy to show in class and easy to extend.
"""

CATEGORIES = [
    "Food & Dining",
    "Groceries",
    "Travel & Transport",
    "Shopping",
    "Bills & Utilities",
    "Mobile Recharge",
    "Education",
    "Entertainment",
    "Health & Fitness",
    "Transfers to People",
]

INCOME_CATEGORY = "Income / Received"

# Merchants used to build the labelled training set (synthetic, Indian context)
MERCHANTS = {
    "Food & Dining": [
        "Swiggy", "Zomato", "Dominos Pizza", "McDonalds", "Starbucks", "Chaayos",
        "Cafe Coffee Day", "Haldiram", "KFC", "Burger King", "Wow Momo", "Subway",
        "Vaishali Restaurant", "Goodluck Cafe", "Pizza Hut", "Barbeque Nation",
        "Third Wave Coffee", "Tea Villa Cafe", "Behrouz Biryani", "Faasos",
        "College Canteen", "Amruttulya Tea",
    ],
    "Groceries": [
        "BigBasket", "Blinkit", "Zepto", "DMart", "Reliance Fresh", "Swiggy Instamart",
        "JioMart", "More Supermarket", "Natures Basket", "Star Bazaar", "Kirana Store",
        "Amul Parlour", "Chitale Dairy", "Spencers",
    ],
    "Travel & Transport": [
        "Uber", "Ola", "Rapido", "IRCTC", "Pune Metro", "MSRTC", "Indian Oil Petrol",
        "HP Petrol Pump", "Bharat Petroleum", "RedBus", "MakeMyTrip", "IndiGo Airlines",
        "FASTag Recharge", "PMPML Bus Pass", "Namma Yatri", "Yulu",
    ],
    "Shopping": [
        "Amazon", "Flipkart", "Myntra", "Ajio", "Meesho", "Nykaa", "Decathlon", "Croma",
        "Reliance Digital", "Lifestyle", "Westside", "Zara", "H&M", "Lenskart",
        "Tata Cliq", "Snitch",
    ],
    "Bills & Utilities": [
        "MSEDCL Electricity", "Mahanagar Gas", "ACT Fibernet", "Airtel Xstream Fiber",
        "Tata Power", "PMC Water Tax", "Bharat Gas LPG", "JioFiber",
        "Society Maintenance", "Indane Gas", "BESCOM", "Hathway Broadband",
    ],
    "Mobile Recharge": [
        "Jio Prepaid Recharge", "Airtel Prepaid", "Vi Recharge", "BSNL Recharge",
        "Jio Recharge", "Airtel Postpaid Bill", "Tata Play DTH", "Dish TV Recharge",
    ],
    "Education": [
        "Vishwakarma University Fees", "Coursera", "Udemy", "Unacademy", "Byjus",
        "Physics Wallah", "Exam Fee Portal", "Crossword Bookstore", "GeeksforGeeks",
        "Scaler Academy", "NPTEL Exam Fee", "Xerox Centre", "Stationery Mart",
        "Pragati Book Depot",
    ],
    "Entertainment": [
        "Netflix", "Spotify", "Amazon Prime", "Disney Hotstar", "BookMyShow",
        "PVR Cinemas", "INOX", "YouTube Premium", "Steam Games", "JioCinema",
        "SonyLIV", "Google Play",
    ],
    "Health & Fitness": [
        "Apollo Pharmacy", "PharmEasy", "Tata 1mg", "Netmeds", "Cult.fit", "Golds Gym",
        "Practo", "MedPlus", "Dr Lal PathLabs", "Ruby Hall Clinic", "Fitness First Gym",
        "Wellness Forever",
    ],
}

# Small local shops: the name itself says what they sell. Mixing these in
# teaches the model words like "medical", "kirana", "petrol", not just brands.
LOCAL_PREFIXES = ["Shree", "Om Sai", "Balaji", "Ganesh", "Laxmi", "Jai Bhavani", "New",
                  "Royal", "Krishna", "Siddhivinayak", "Mauli", "Swami Samarth"]
LOCAL_SUFFIXES = {
    "Food & Dining": ["Restaurant", "Cafe", "Snacks Centre", "Vada Pav", "Bhojanalay", "Misal House", "Sweets"],
    "Groceries": ["Kirana Stores", "General Stores", "Supermarket", "Dairy", "Vegetables", "Provision Store"],
    "Travel & Transport": ["Petrol Pump", "Auto Service", "Travels", "Fuels", "Tours", "Cab Service"],
    "Shopping": ["Garments", "Fashion", "Mobile Shop", "Footwear", "Electronics", "Collection"],
    "Bills & Utilities": ["Gas Agency", "Electricity Bill", "Broadband", "Water Supply", "Cable Network"],
    "Mobile Recharge": ["Mobile Recharge", "Telecom", "Recharge Point", "DTH Recharge"],
    "Education": ["Book Depot", "Xerox", "Stationers", "Classes", "Academy", "Tuitions"],
    "Entertainment": ["Cinema", "Gaming Zone", "Theatre", "Bowling", "Multiplex"],
    "Health & Fitness": ["Medical", "Chemist", "Clinic", "Hospital", "Diagnostics", "Gym"],
}

FIRST_NAMES = [
    "Rahul", "Priya", "Aditya", "Sneha", "Rohan", "Ananya", "Vikram", "Pooja",
    "Omkar", "Sakshi", "Yash", "Shruti", "Aniket", "Neha", "Siddharth", "Tanvi",
    "Kunal", "Isha", "Pranav", "Riya", "Harsh", "Mrunal", "Tejas", "Gauri",
    "Akash", "Divya", "Saurabh", "Komal", "Nikhil", "Aarti",
]
LAST_NAMES = [
    "Sharma", "Patil", "Deshmukh", "Joshi", "Kulkarni", "Gupta", "Verma", "Shinde",
    "Pawar", "Jadhav", "Mehta", "Iyer", "Reddy", "Nair", "Chavan", "More",
    "Bhosale", "Gaikwad", "Singh", "Agarwal",
]

UPI_HANDLES_MERCHANT = ["@ybl", "@paytm", "@axl", "@icici", "@hdfcbank", "@yesbank", "@okbizaxis"]
UPI_HANDLES_PERSON = ["@okhdfcbank", "@oksbi", "@okaxis", "@okicici", "@ybl", "@paytm", "@ibl"]

# Rule-based fallback: if the ML model is unsure, these keywords decide.
KEYWORD_RULES = {
    "Food & Dining": ["swiggy", "zomato", "pizza", "cafe", "coffee", "restaurant", "biryani",
                      "momo", "burger", "kfc", "mcdonald", "tea", "canteen", "dhaba", "food", "bakery", "hotel", "vada pav", "misal", "sweets"],
    "Groceries": ["bigbasket", "blinkit", "zepto", "dmart", "mart", "fresh", "kirana",
                  "supermarket", "instamart", "dairy", "grocery", "vegetable"],
    "Travel & Transport": ["uber", "ola", "rapido", "irctc", "metro", "petrol", "fuel", "bus",
                           "redbus", "makemytrip", "indigo", "fastag", "railway", "cab", "auto"],
    "Shopping": ["amazon", "flipkart", "myntra", "ajio", "meesho", "nykaa", "decathlon",
                 "croma", "lifestyle", "store", "fashion", "online", "deals"],
    "Bills & Utilities": ["electricity", "msedcl", "gas", "fibernet", "broadband", "water",
                          "power", "maintenance", "bill", "lpg"],
    "Mobile Recharge": ["recharge", "prepaid", "postpaid", "jio", "airtel", "vodafone", "bsnl", "dth"],
    "Education": ["university", "college", "fees", "coursera", "udemy", "unacademy", "exam",
                  "book", "xerox", "stationery", "academy", "course", "tuition"],
    "Entertainment": ["netflix", "spotify", "prime", "hotstar", "bookmyshow", "pvr", "inox",
                      "youtube", "steam", "movie", "cinema", "game"],
    "Health & Fitness": ["pharmacy", "pharmeasy", "1mg", "netmeds", "gym", "cult", "clinic",
                         "hospital", "medical", "medplus", "lab", "doctor", "health"],
}

# Common words seen in UPI scam payee names (fraud-awareness, FLB Unit 3)
SCAM_WORDS = ["kyc", "refund", "lottery", "reward", "verify", "prize", "cashback-claim",
              "helpdesk", "customer-care", "kbc", "update-help"]

# 50/30/20 rule: which categories are Needs and which are Wants
NEEDS = ["Groceries", "Travel & Transport", "Bills & Utilities", "Mobile Recharge",
         "Education", "Health & Fitness"]
WANTS = ["Food & Dining", "Shopping", "Entertainment", "Transfers to People"]
