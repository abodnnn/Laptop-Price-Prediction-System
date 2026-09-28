<div align="center">

# 💻 نظام التنبؤ بأسعار أجهزة اللابتوب

### Laptop Price Prediction System

<br>

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![CustomTkinter](https://img.shields.io/badge/CustomTkinter-5.0+-00C4CC?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

<br>

**تطبيق سطح مكتب احترافي للتنبؤ بأسعار أجهزة اللابتوب بالدولار الأمريكي**
**باستخدام تعلم الآلة (Machine Learning) وواجهة رسومية عصرية**


<br>

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- ██████╗  English Section (LTR)  ██████╗                    -->
<!-- ═══════════════════════════════════════════════════════════ -->

<div dir="ltr" align="left">

# 📖 Project Description

## 🔭 Overview

The **Laptop Price Prediction System** is a full-featured **desktop application** built with **Python** that combines the power of **Machine Learning** with an elegant **modern GUI** to predict laptop prices in US Dollars.

The application is trained on a **real-world dataset** scraped from **Amazon**, containing data on over **4,400 laptops** with varying specifications and prices. It uses a **Random Forest Regressor** from **Scikit-Learn** to deliver intelligent price predictions.

<br>

## 🎯 Problem & Solution

### The Problem
When purchasing a new laptop, users often struggle to estimate a fair price based on desired specifications. Prices vary significantly across manufacturers and depend on multiple factors such as processor, RAM, storage, graphics card, and more.

### The Solution
An **intelligent machine learning model** trained on real Amazon data that can predict the price of any laptop based on its technical specifications. The model is wrapped in a **sleek, modern dashboard GUI** that allows anyone to get instant price estimates.

### Problem Type

| Item | Detail |
|:---|:---|
| **Problem Type** | Regression — predicting a continuous numerical value |
| **Target Variable** | `Price_USD` — laptop price in US Dollars |
| **Number of Features** | 8 features: 4 Categorical + 4 Numerical |
| **Algorithm** | Random Forest Regressor |
| **Data Source** | Amazon Laptop Prices Dataset |

<br>

## ✨ Key Features

### 🧠 Machine Learning Engine
- ✅ **Random Forest** model with 300 decision trees for high-accuracy predictions
- ✅ Advanced data preprocessing with intelligent feature engineering
- ✅ Integrated **Pipeline** ensuring no data leakage
- ✅ Automatic model save/load using `joblib` serialization
- ✅ Three performance metrics: **R² Score**, **MAE**, and **RMSE**

### 🎨 User Interface
- ✅ Modern **Dashboard-style** design with professional aesthetics
- ✅ **Dark Mode** and **Light Mode** with instant switching
- ✅ Full **RTL (Right-to-Left)** support for Arabic text
- ✅ Smart dropdown menus auto-populated from real data
- ✅ Automatic numeric input validation
- ✅ Instant price display in professionally formatted USD

### ⚡ Performance & UX
- ✅ **Asynchronous** model and data loading — no UI freezing
- ✅ In-app **model retraining** capability
- ✅ Quick **reset** for all input fields
- ✅ Clear, descriptive error messages

<br>

## 📊 Dataset

### General Information

| Item | Detail |
|:---|:---|
| **Name** | Amazon Laptop Prices v01 |
| **File** | `amazon_laptop_prices_v01.csv` |
| **Records** | 4,447 laptops |
| **Raw Columns** | 14 columns |
| **Size** | ~540 KB |
| **Source** | Laptop prices and specifications from Amazon |

### Raw Data Columns

| Column | Description | Example |
|:---|:---|:---|
| `brand` | Manufacturer | HP, Dell, Apple |
| `model` | Model name | MacBook Air |
| `screen_size` | Screen size (text) | "15.6 Inches" |
| `color` | Color | Silver, Black |
| `harddisk` | Storage capacity (text) | "512 GB", "1 TB" |
| `cpu` | Processor (text) | "Intel Core i7" |
| `ram` | RAM (text) | "8 GB" |
| `OS` | Operating System | "Windows 11 Pro" |
| `special_features` | Special features | "Backlit Keyboard" |
| `graphics` | Graphics card (text) | "Integrated" |
| `graphics_coprocessor` | GPU coprocessor | Intel, NVIDIA |
| `cpu_speed` | CPU speed (text) | "2.4 GHz" |
| `rating` | User rating | 4.5 |
| `price` | Price (text with $) | "$1,599.00" |

### Final Features After Preprocessing

```
Categorical Features:                  Numerical Features:
┌─────────────────────┐               ┌──────────────────────┐
│  brand              │               │  Screen_Inches       │
│  Cpu_Category       │               │  Ram_GB              │
│  Graphics_Type      │               │  Storage_GB          │
│  OS_Category        │               │  Cpu_Speed_GHz       │
└─────────────────────┘               └──────────────────────┘
         ↓                                      ↓
   OneHotEncoder                          StandardScaler
         ↓                                      ↓
         └──────── ColumnTransformer ───────────┘
                          ↓
               RandomForestRegressor
                          ↓
                  Price_USD (Prediction)
```

<br>

## 🔬 Data Preprocessing

The data went through several meticulous processing stages before training:

### 1️⃣ Numeric Column Cleaning
| Function | Task | Example |
|:---|:---|:---|
| `_parse_price()` | Convert price text to float | `"$1,599.00"` → `1599.00` |
| `_parse_ram()` | Extract RAM in GB | `"8 GB"` → `8.0`, `"64 MB"` → `0.064` |
| `_parse_storage()` | Extract storage in GB | `"1 TB"` → `1000`, `"512 GB"` → `512` |
| `_parse_screen_size()` | Extract screen size in inches | `"15.6 Inches"` → `15.6` |
| `_parse_cpu_speed()` | Extract CPU speed in GHz | `"2.4 GHz"` → `2.4` |

### 2️⃣ Categorical Feature Engineering
- **CPU Classification** → 16 main categories (Intel Core i3/i5/i7/i9, AMD Ryzen, Apple Silicon, ...)
- **Graphics Classification** → 12 categories (NVIDIA RTX/GTX, AMD Radeon, Intel Integrated, ...)
- **OS Classification** → 10 categories (Windows 10/11, macOS, Chrome OS, Linux, ...)

### 3️⃣ Additional Cleaning
- Unified manufacturer names (e.g., `ASUS` → `Asus`, `DELL` → `Dell`)
- Removed records with missing values in essential columns
- Removed illogical values (price ≤ 0, screen < 5" or > 25")
- Filled missing CPU speed with median (more robust than mean against outliers)
- Grouped rare brands (< 5 devices) under "Other"

<br>

## 🤖 Model Details

### Algorithm: Random Forest Regressor

| Hyperparameter | Value | Description |
|:---|:---|:---|
| `n_estimators` | 300 | Number of decision trees in the forest |
| `max_depth` | 20 | Maximum depth per tree |
| `min_samples_split` | 5 | Minimum samples required to split a node |
| `random_state` | 42 | Fixed seed for reproducibility |
| `n_jobs` | -1 | Use all available CPU cores |

### Why Random Forest?
- ✅ Handles **non-linear relationships** in laptop pricing
- ✅ Works efficiently with **mixed feature types** (categorical + numerical)
- ✅ **Resistant to overfitting** through ensemble of 300 trees
- ✅ No **distributional assumptions** about the data
- ✅ Supports **parallel training** for faster performance

### Performance Metrics

| Metric | Description |
|:---|:---|
| **R² Score** | Proportion of variance explained by the model (closer to 1.0 = better) |
| **MAE** | Mean Absolute Error in USD — average prediction deviation |
| **RMSE** | Root Mean Squared Error — penalizes large errors more heavily |

### Data Split
- **80%** Training Set
- **20%** Test Set
- Fixed random seed (`random_state=42`) for reproducibility

<br>

## 🏗️ Project Structure

```
Project_ML/
│
├── 📄 laptop_price_predictor.py      ← Main source code (1,411 lines)
│       Contains:
│       • LaptopPriceModel  → ML Engine class
│       • LaptopPriceDashboard → GUI Dashboard class
│
├── 📊 amazon_laptop_prices_v01.csv   ← Raw dataset (~4,447 records)
│
├── 🤖 laptop_model.joblib            ← Trained model file (~21 MB)
│
├── 📘 README.md                      ← Documentation (this file)
│
└── 📘 Project_Discussion_Guide.md    ← Academic discussion guide
```

### File Descriptions

| File | Role | Details |
|:---|:---|:---|
| `laptop_price_predictor.py` | Main source code | Contains all logic: data processing, model training, and GUI design. Single monolithic file (~66 KB) |
| `amazon_laptop_prices_v01.csv` | Dataset | Raw CSV data with 4,447 laptop entries from Amazon |
| `laptop_model.joblib` | Saved model | Pre-trained ML model in joblib format. Auto-loaded at startup to skip retraining |
| `Project_Discussion_Guide.md` | Discussion guide | Comprehensive academic guide with detailed code explanations and expected Q&A |

<br>

## 🖥️ GUI Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                 Main Window (1280×780)                        │
│ ┌──────────┬──────────────────────────────────────────────┐  │
│ │          │                                              │  │
│ │ Sidebar  │      🔮 Price Prediction Dashboard           │  │
│ │          │  ┌──────────────────────────────────────┐    │  │
│ │ 💻 Logo  │  │                                      │    │  │
│ │          │  │   📋 Device Specs (8 fields)          │    │  │
│ │ ✅ Status│  │   [4 Dropdowns]  [4 Numeric Inputs]  │    │  │
│ │          │  │                                      │    │  │
│ │ ☀️ Theme │  └──────────────────────────────────────┘    │  │
│ │ 🔄 Reset │  ┌──────────────────────────────────────┐    │  │
│ │ 🧠 Train │  │   🚀 Predict Price Now                │    │  │
│ │          │  └──────────────────────────────────────┘    │  │
│ │          │  ┌──────────────────────────────────────┐    │  │
│ │          │  │   💰 Predicted Price: $X,XXX.XX       │    │  │
│ │          │  └──────────────────────────────────────┘    │  │
│ │ Powered  │  ┌──────────┬──────────┬──────────┐         │  │
│ │ by Devs  │  │ R² Score │   MAE   │   RMSE   │         │  │
│ └──────────┴──┴──────────┴──────────┴──────────┘         │  │
└──────────────────────────────────────────────────────────────┘
```

<br>

## 🚀 Quick Start

### Prerequisites

| Library | Purpose |
|:---|:---|
| **Python 3.8+** | Core programming language |
| **pandas** | Data manipulation and analysis |
| **numpy** | Numerical computations |
| **scikit-learn** | Machine learning algorithms |
| **joblib** | Model serialization |
| **customtkinter** | Modern GUI toolkit |

### Installation & Running

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Laptop-Price-Predictor.git
cd Laptop-Price-Predictor

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux/Mac

# 3. Install required libraries
pip install pandas numpy scikit-learn joblib customtkinter

# 4. Run the application
python laptop_price_predictor.py
```

### Application Startup Behavior

```
[Launch] → Does laptop_model.joblib exist?
               │
         ╔═ YES ══╗  Loads saved model (takes seconds)
         ║        ║  → Model ready for predictions ✅
         ╚════════╝
               │
         ╔═ NO ═══╗  Trains a new model (15-60 seconds)
         ║        ║  → Auto-saves for future launches
         ╚════════╝
```

<br>

## 🛠️ Tech Stack

| Technology | Version | Role |
|:---|:---|:---|
| **Python** | 3.8+ | Core programming language |
| **Scikit-Learn** | 1.0+ | ML model building and training |
| **Pandas** | 2.0+ | Tabular data processing |
| **NumPy** | 1.20+ | Array mathematical operations |
| **CustomTkinter** | 5.0+ | Modern GUI framework |
| **Joblib** | 1.0+ | Model serialization and persistence |
| **Threading** | Built-in | Asynchronous loading |
| **Regular Expressions** | Built-in | Numeric data extraction from text |

<br>

## 🔮 Future Improvements

- 📈 Add **interactive charts** for price distribution and feature importance visualization
- 🔍 Experiment with stronger algorithms like **XGBoost** and **LightGBM**
- 📊 Implement **Cross-Validation** for more reliable evaluation
- 🧪 Add **GridSearchCV** for automatic hyperparameter tuning
- 📄 Export results as **PDF** or **CSV** files
- 🔗 Convert to a **web application** using Flask or Streamlit
- 📦 Refactor code into modular files (`model.py`, `ui.py`, `utils.py`)
- 🧹 Add **unit tests** for cleaning and prediction functions

<br>

## 👨‍💻 Developers

<div align="center">

**Machine Learning** Course Project

**Developed by:** Abdullah & Osama

**Version:** v2.0 | **Data Source:** Amazon Dataset

</div>

<br>

## 📄 License

This project is licensed under the [MIT License](LICENSE) — feel free to use, modify, and distribute it.

</div>

<br>

---

<div align="center">

**⭐ إذا أعجبك المشروع، لا تنسَ تقييمه بنجمة! | If you like this project, don't forget to give it a star! ⭐**

</div>
