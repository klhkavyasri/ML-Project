import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/placement_model.pkl")

# Load label encoders
label_encoders = joblib.load("models/label_encoders.pkl")


def predict_placement(data):

    # Create input using the SAME columns used during training
    input_data = pd.DataFrame([{
        "Gender": data["Gender"],
        "City": data["City"],
        "CollegeTier": data["CollegeTier"],
        "Stream": data["Stream"],
        "Specialisation": data["Specialisation"],
        "Hostel": data["Hostel"],
        "HistoryOfBacklogs": data["HistoryOfBacklogs"],

        "SGPA_Sem1": float(data["SGPA_Sem1"]),
        "SGPA_Sem2": float(data["SGPA_Sem2"]),
        "SGPA_Sem3": float(data["SGPA_Sem3"]),
        "SGPA_Sem4": float(data["SGPA_Sem4"]),
        "SGPA_Sem5": float(data["SGPA_Sem5"]),
        "SGPA_Sem6": float(data["SGPA_Sem6"]),
        "SGPA_Sem7": float(data["SGPA_Sem7"]),
        "SGPA_Sem8": float(data["SGPA_Sem8"]),

        "CGPA": float(data["CGPA"]),
        "AttendancePercent": float(data["AttendancePercent"]),

        "Internships": int(data["Internships"]),
        "Projects": int(data["Projects"]),
        "Workshops": int(data["Workshops"]),
        "Certifications": int(data["Certifications"]),
        "Publications": int(data["Publications"]),

        "AptitudeTestScore": float(data["AptitudeTestScore"]),
        "SoftSkillsRating": float(data["SoftSkillsRating"]),
        "CodingTestScore": float(data["CodingTestScore"]),
        "MockInterviewScore": float(data["MockInterviewScore"]),

        "ExtraCurricular": data["ExtraCurricular"],

        # These are calculated/defaulted because they are not directly entered
        "CGPA_Tier": "High",
        "IsAnomaly": 0
    }])

    # Convert categorical columns using the same encoders
    for column, encoder in label_encoders.items():
        if column in input_data.columns:
            input_data[column] = encoder.transform(
                input_data[column].astype(str)
            )

    # Make prediction
    prediction = model.predict(input_data)[0]

    return prediction