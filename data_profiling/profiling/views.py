import os
import pandas as pd
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.urls import reverse
import chardet  # For automatic encoding detection
import google.generativeai as genai  # Gemini AI
import matplotlib

matplotlib.use("Agg")  # Use a non-GUI backend
import matplotlib.pyplot as plt
from .models import UploadedFile

# Configure Gemini API Key
GEMINI_API_KEY = getattr(settings, "GOOGLE_API_KEY", None)
if not GEMINI_API_KEY:
    raise ValueError("⚠️ GOOGLE_API_KEY is not set in settings.py")
genai.configure(api_key=GEMINI_API_KEY)

# Settings for file upload directory
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Home Page
def index(request):
    return render(request, "index.html")


def home(request):
    return render(request, "home.html")


# File Upload Function
def upload_file(request):
    if request.method == "POST" and request.FILES.get("dataset"):
        dataset = request.FILES["dataset"]
        fs = FileSystemStorage(location=UPLOAD_DIR)
        file_path = fs.save(dataset.name, dataset)
        file_full_path = os.path.join(UPLOAD_DIR, file_path)

        # Process the uploaded file
        profiling_results = process_dataset(file_full_path)

        # Remove the uploaded file after processing
        os.remove(file_full_path)

        return render(request, "results.html", {"results": profiling_results})

    return HttpResponse("No file uploaded.")


# Function to Detect File Encoding
def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        result = chardet.detect(
            f.read(100000)
        )  # Read first 100k bytes for encoding detection
    return result["encoding"]


# Dataset Processing Function
def process_dataset(file_path):
    try:
        encoding = detect_encoding(file_path)  # Detect encoding

        # Read CSV or Excel file with the detected encoding
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, encoding=encoding, on_bad_lines="skip")
            file_ext = "csv"
        elif file_path.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
            file_ext = "excel"
        else:
            return {"error": "Unsupported file format."}

        # Initial Stats
        total_rows, total_cols = df.shape

        # Data Cleaning
        errors = detect_errors(df)
        duplicate_count = df.duplicated().sum()
        df_cleaned = remove_duplicates(df)
        missing_values_count = df_cleaned.isnull().sum().sum()
        df_cleaned = handle_missing_values(df_cleaned)
        df_cleaned = standardize_data(df_cleaned)

        # Save Cleaned File
        cleaned_filename = "cleaned_" + os.path.basename(file_path)
        cleaned_path = os.path.join(UPLOAD_DIR, cleaned_filename)
        if file_ext == "csv":
            df_cleaned.to_csv(cleaned_path, index=False, encoding="utf-8")
        else:
            df_cleaned.to_excel(cleaned_path, index=False)

        # Generate Report
        report = f"""
        DATA CLEANING REPORT

        Total Rows: {total_rows}
        Total Columns: {total_cols}
        Errors Detected: {len(errors)} columns had issues
        Duplicates Removed: {duplicate_count}
        Missing Values Filled: {missing_values_count}
        Standardization Applied: Dates and text formatting standardized
        """

        # Save report to a text file
        report_filename = "cleaning_report.txt"
        report_path = os.path.join(UPLOAD_DIR, report_filename)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        # Store file reference in database
        UploadedFile.objects.create(file=cleaned_filename)

        # Profiling results
        profiling_results = {
            "null_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "statistics": df.describe(include="all").to_dict(),
            "categories": categorize_columns(df),
            "errors": errors,
            "cleaned_file": cleaned_filename,
            "report": report,
        }

        return profiling_results
    except Exception as e:
        return {"error": str(e)}


# Error Detection
def detect_errors(df):
    errors = {}
    for col in df.columns:
        if df[col].dtype == "object":  # Text columns
            errors[col] = df[col][
                df[col].str.contains(r"[^a-zA-Z0-9\s]", na=False)
            ].tolist()
        elif df[col].dtype in ["int64", "float64"]:  # Numeric columns
            errors[col] = df[col][df[col] < 0].tolist()  # Detect negative values
    return errors


# Remove Duplicates
def remove_duplicates(df):
    return df.drop_duplicates()


# Handle Missing Values
def handle_missing_values(df):
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col].fillna(df[col].mean(), inplace=True)  # Fill with mean
        else:
            df[col].fillna("Unknown", inplace=True)  # Fill text with "Unknown"
    return df


# Standardization (Fix Date Formats, Capitalization)
def standardize_data(df):
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )  # Convert to YYYY-MM-DD
        elif df[col].dtype == "object":
            df[col] = df[col].str.strip().str.title()  # Capitalize words
    return df


# Function to Categorize Columns
def categorize_columns(df):
    categories = {}
    for column in df.columns:
        dtype = df[column].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            categories[column] = "Numerical"
        elif pd.api.types.is_object_dtype(dtype):
            categories[column] = "Categorical"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            categories[column] = "DateTime"
        else:
            categories[column] = "Other"
    return categories


# Data Bot (AI Chat Assistant)
def databot_result(request):
    message = request.GET.get("message", "")
    return render(request, "databot_result.html", {"message": message})


# File Upload for Data Bot
def upload_dataset(request):
    if request.method == "POST" and request.FILES.get("dataset"):
        dataset = request.FILES["dataset"]
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(dataset.name, dataset)
        dataset_path = os.path.join(settings.MEDIA_ROOT, filename)

        request.session["dataset_path"] = dataset_path
        request.session.modified = True

        return redirect(
            f"{reverse('databot_result')}?message=Dataset uploaded successfully!"
        )

    return render(request, "databot_result.html")


# Function to Answer Questions from the Dataset
def ask_question(request):
    dataset_path = request.session.get("dataset_path")

    if request.method == "POST" and dataset_path:
        question = request.POST.get("question")
        if not question:
            return render(
                request, "databot_result.html", {"error": "Please enter a question"}
            )

        try:
            encoding = detect_encoding(dataset_path)
            df = pd.read_csv(dataset_path, encoding=encoding, on_bad_lines="skip")

            df_json = df.head(5).to_dict(orient="records")

            prompt = f"""
            You are an AI analyzing a dataset. The dataset contains these records:

            {df_json}

            Answer the user's question based on this dataset.

            User's Question: {question}
            """

            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            response = model.generate_content(prompt)
            answer_text = (
                response.text
                if hasattr(response, "text")
                else "⚠️ No response generated."
            )

            return render(request, "databot_result.html", {"answer": answer_text})

        except Exception as e:
            return render(
                request,
                "databot_result.html",
                {"error": f"Error processing dataset: {str(e)}"},
            )

    return render(request, "databot_result.html")


# Data Visualization
def datavis_result(request):
    return render(request, "datavis_result.html")


def uploading_file(request):
    if request.method == "POST" and request.FILES.get("file"):
        file = request.FILES["file"]
        file_path = os.path.join(settings.MEDIA_ROOT, file.name)

        # Save file to media directory
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        with open(file_path, "wb+") as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        # Read file based on extension
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file.name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(file_path)
            else:
                return HttpResponse("Unsupported file format", status=400)

            # Separate categorical and numerical columns
            categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
            numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

            return render(
                request,
                "datavis_result.html",
                {
                    "file_path": file_path,
                    "categorical_cols": categorical_cols,
                    "numerical_cols": numerical_cols,
                },
            )
        except Exception as e:
            return HttpResponse(f"Error reading file: {e}", status=500)

    return render(request, "datavis_result.html")


def generate_chart(request):
    if request.method == "POST":
        file_path = request.POST.get("file_path")
        x_col = request.POST.get("x_col")
        y_col = request.POST.get("y_col")
        chart_type = request.POST.get("chart_type")

        # Ensure the file exists
        if not os.path.exists(file_path):
            return HttpResponse("Error: File not found", status=404)

        try:
            # Load file based on extension
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file_path.endswith((".xls", ".xlsx")):
                df = pd.read_excel(file_path)
            else:
                return HttpResponse("Unsupported file format", status=400)

            # Ensure selected columns exist in DataFrame
            if x_col not in df.columns or y_col not in df.columns:
                return HttpResponse("Error: Invalid column selection", status=400)

            df = df.dropna(subset=[y_col])
            plt.figure(figsize=(8, 5))

            # Generate different chart types
            if chart_type == "bar":
                plt.bar(df[x_col], df[y_col], color="blue")
            elif chart_type == "line":
                plt.plot(df[x_col], df[y_col], marker="o", linestyle="-", color="green")
            elif chart_type == "scatter":
                plt.scatter(df[x_col], df[y_col], color="red", alpha=0.7)
            elif chart_type == "pie":
                df = df[df[y_col] > 0]
                df[y_col] = df[y_col].astype(float)
                plt.pie(df[y_col], labels=df[x_col], autopct="%1.1f%%", startangle=90)
                plt.axis("equal")

            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"{y_col} vs {x_col}")

            # Save chart in media folder
            chart_path = os.path.join(settings.MEDIA_ROOT, "chart.png")
            plt.savefig(chart_path)
            plt.close()

            return render(
                request,
                "datavis_result.html",
                {
                    "img_url": settings.MEDIA_URL + "chart.png",
                    "file_path": file_path,
                    "categorical_cols": df.select_dtypes(
                        include=["object"]
                    ).columns.tolist(),
                    "numerical_cols": df.select_dtypes(
                        include=["number"]
                    ).columns.tolist(),
                },
            )

        except Exception as e:
            return HttpResponse(f"Error generating chart: {e}", status=500)

    return HttpResponse("Invalid request", status=400)
