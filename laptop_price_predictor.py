# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║        تطبيق التنبؤ بأسعار أجهزة اللابتوب                   ║
║        Laptop Price Prediction Application                   ║
║        مبني بـ Python + CustomTkinter + Scikit-Learn         ║
║                                                              ║
║        الداتا سيت: Amazon Laptop Prices v01                  ║
╚══════════════════════════════════════════════════════════════╝

الوصف: تطبيق سطح مكتب احترافي للتنبؤ بأسعار اللابتوب
       باستخدام نموذج Random Forest Regressor
       مع واجهة عصرية على شكل لوحة تحكم (Dashboard)
"""

# ══════════════════════════════════════════════════════════════
# استيراد المكتبات المطلوبة
# ══════════════════════════════════════════════════════════════
import os
import re 
import sys
import threading
import numpy as np
import pandas as pd
import joblib
import customtkinter as ctk
from tkinter import messagebox

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ══════════════════════════════════════════════════════════════
# ثوابت التصميم - ألوان ونمط لوحة التحكم
# ══════════════════════════════════════════════════════════════

# ألوان الوضع الداكن (Dark Mode)
DARK = {
    "bg_primary":    "#0f0f1a",       # خلفية رئيسية - أسود مزرق عميق
    "bg_secondary":  "#1a1a2e",       # خلفية ثانوية - بنفسجي داكن
    "bg_card":       "#16213e",       # خلفية البطاقات
    "bg_sidebar":    "#0a0a14",       # خلفية القائمة الجانبية
    "bg_input":      "#1e2a4a",       # خلفية حقول الإدخال
    "accent":        "#6c63ff",       # لون مميز - بنفسجي
    "accent_hover":  "#5a52d5",       # لون مميز عند التحويم
    "accent_glow":   "#7b73ff",       # توهج اللون المميز
    "success":       "#00d9a6",       # أخضر فيروزي للنجاح
    "warning":       "#f7b731",       # أصفر ذهبي للتحذير
    "danger":        "#ff6b6b",       # أحمر ناعم للخطأ
    "text_primary":  "#e8e8f0",       # نص رئيسي
    "text_secondary":"#8888aa",       # نص ثانوي
    "text_muted":    "#555577",       # نص خافت
    "border":        "#2a2a4e",       # لون الحدود
    "result_bg":     "#1a2847",       # خلفية بطاقة النتيجة
    "metric_bg":     "#141428",       # خلفية بطاقة المقاييس
}

# ألوان الوضع الفاتح (Light Mode)
LIGHT = {
    "bg_primary":    "#f0f2f5",
    "bg_secondary":  "#ffffff",
    "bg_card":       "#ffffff",
    "bg_sidebar":    "#e8eaf0",
    "bg_input":      "#f5f5fa",
    "accent":        "#6c63ff",
    "accent_hover":  "#5a52d5",
    "accent_glow":   "#8b83ff",
    "success":       "#00b894",
    "warning":       "#e17055",
    "danger":        "#d63031",
    "text_primary":  "#1a1a2e",
    "text_secondary":"#636e72",
    "text_muted":    "#b2bec3",
    "border":        "#dfe6e9",
    "result_bg":     "#eef0ff",
    "metric_bg":     "#f8f9ff",
}


# ══════════════════════════════════════════════════════════════
# الفئة الأولى: محرك تعلم الآلة (ML Engine)
# ══════════════════════════════════════════════════════════════

class LaptopPriceModel:
    """
    فئة محرك تعلم الآلة - المسؤولة عن:
    - تحميل وتنظيف بيانات أمازون من ملف CSV
    - بناء وتدريب نموذج التنبؤ (RandomForestRegressor)
    - حفظ وتحميل النموذج المدرب (joblib)
    - تنفيذ عمليات التنبؤ وحساب مقاييس الأداء
    """

    # مسار حفظ النموذج المدرب
    MODEL_PATH = "laptop_model.joblib"
    # مسار ملف البيانات الجديد (أمازون)
    DATA_PATH = "amazon_laptop_prices_v01.csv"

    def __init__(self):
        """تهيئة المحرك: تحميل البيانات ثم تحميل أو تدريب النموذج"""
        self.df = None              # إطار البيانات الأصلي
        self.df_clean = None        # إطار البيانات بعد التنظيف
        self.pipeline = None        # خط أنابيب التعلم الآلي
        self.metrics = {}           # مقاييس أداء النموذج
        self.is_trained = False     # هل النموذج مدرب؟

        # الأعمدة الفئوية (النصية) المستخدمة كميزات
        self.cat_features = ["brand", "Cpu_Category", "Graphics_Type", "OS_Category"]
        # الأعمدة الرقمية المستخدمة كميزات
        self.num_features = ["Screen_Inches", "Ram_GB", "Storage_GB", "Cpu_Speed_GHz"]

    # ─────────────────────────────────────────────────────────
    # دوال مساعدة لتنظيف البيانات
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_price(val):
        """
        تحويل السعر من نص إلى رقم عشري
        مثال: "$1,599.00 " → 1599.00
        """
        if pd.isna(val):
            return np.nan
        s = str(val).replace("$", "").replace(",", "").strip()
        try:
            return float(s)
        except ValueError:
            return np.nan

    @staticmethod
    def _parse_ram(val):
        """
        استخراج سعة الرام بالجيجابايت من النص
        مثال: "8 GB" → 8.0 ، "64 MB" → 0.064
        """
        if pd.isna(val):
            return np.nan
        s = str(val).strip().upper()
        # استخراج الرقم
        match = re.search(r"([\d.]+)", s)
        if not match:
            return np.nan
        num = float(match.group(1))
        # تحويل MB إلى GB
        if "MB" in s:
            return num / 1024.0
        return num

    @staticmethod
    def _parse_storage(val):
        """
        استخراج سعة التخزين بالجيجابايت من النص
        مثال: "512 GB" → 512 ، "1 TB" → 1000 ، "128 MB" → 0.125
        """
        if pd.isna(val):
            return np.nan
        s = str(val).strip().upper()
        match = re.search(r"([\d.]+)", s)
        if not match:
            return np.nan
        num = float(match.group(1))
        if "TB" in s:
            return num * 1000.0
        elif "MB" in s:
            return num / 1024.0
        return num  # GB افتراضي

    @staticmethod
    def _parse_screen_size(val):
        """
        استخراج حجم الشاشة بالبوصة من النص
        مثال: "15.6 Inches" → 15.6
        """
        if pd.isna(val):
            return np.nan
        match = re.search(r"([\d.]+)", str(val))
        if match:
            return float(match.group(1))
        return np.nan

    @staticmethod
    def _parse_cpu_speed(val):
        """
        استخراج سرعة المعالج بالجيجاهرتز من النص
        مثال: "2.4 GHz" → 2.4 ، "3.4" → 3.4
        """
        if pd.isna(val):
            return np.nan
        match = re.search(r"([\d.]+)", str(val))
        if match:
            return float(match.group(1))
        return np.nan

    @staticmethod
    def _categorize_cpu(val):
        """
        تصنيف المعالج إلى فئات رئيسية مبسطة
        لتقليل عدد الفئات الفريدة وتحسين أداء النموذج
        """
        if pd.isna(val):
            return "Other"
        s = str(val).lower()
        if "i9" in s or "core i9" in s:
            return "Intel Core i9"
        elif "i7" in s or "core i7" in s:
            return "Intel Core i7"
        elif "i5" in s or "core i5" in s:
            return "Intel Core i5"
        elif "i3" in s or "core i3" in s:
            return "Intel Core i3"
        elif "xeon" in s:
            return "Intel Xeon"
        elif "celeron" in s:
            return "Intel Celeron"
        elif "pentium" in s:
            return "Intel Pentium"
        elif "atom" in s:
            return "Intel Atom"
        elif "m1" in s or "m2" in s or "apple" in s:
            return "Apple Silicon"
        elif "ryzen 9" in s:
            return "AMD Ryzen 9"
        elif "ryzen 7" in s:
            return "AMD Ryzen 7"
        elif "ryzen 5" in s:
            return "AMD Ryzen 5"
        elif "ryzen 3" in s:
            return "AMD Ryzen 3"
        elif "ryzen" in s or "amd" in s or "athlon" in s:
            return "AMD Other"
        elif "core m" in s or "core_m" in s:
            return "Intel Core M"
        elif "core 2" in s:
            return "Intel Core 2"
        elif "mediatek" in s or "snapdragon" in s or "cortex" in s or "arm" in s:
            return "ARM/Mobile"
        else:
            return "Other"

    @staticmethod
    def _categorize_graphics(val):
        """
        تصنيف كرت الشاشة إلى فئات رئيسية
        """
        if pd.isna(val):
            return "Unknown"
        s = str(val).lower()
        if "rtx 40" in s:
            return "NVIDIA RTX 40 Series"
        elif "rtx 30" in s:
            return "NVIDIA RTX 30 Series"
        elif "rtx 20" in s or "rtx2" in s:
            return "NVIDIA RTX 20 Series"
        elif "rtx" in s and ("a1" in s or "a2" in s or "a3" in s or "a4" in s or "a5" in s or "ada" in s):
            return "NVIDIA RTX Pro"
        elif "gtx" in s:
            return "NVIDIA GTX"
        elif "quadro" in s:
            return "NVIDIA Quadro"
        elif "nvidia" in s or "geforce" in s:
            return "NVIDIA Other"
        elif "radeon" in s or "amd" in s:
            return "AMD Radeon"
        elif "iris" in s:
            return "Intel Iris"
        elif "intel" in s or "uhd" in s or "hd graphics" in s:
            return "Intel Integrated"
        elif "integrated" in s or "shared" in s:
            return "Integrated"
        elif "dedicated" in s:
            return "Dedicated"
        elif "adreno" in s:
            return "ARM/Mobile GPU"
        else:
            return "Other"

    @staticmethod
    def _categorize_os(val):
        """
        تبسيط أنظمة التشغيل إلى فئات رئيسية
        لأن الداتا سيت تحتوي على عشرات المتغيرات من ويندوز
        """
        if pd.isna(val):
            return "Other"
        s = str(val).lower()
        if "windows 11 pro" in s:
            return "Windows 11 Pro"
        elif "windows 11 home" in s:
            return "Windows 11 Home"
        elif "windows 11" in s:
            return "Windows 11"
        elif "windows 10 pro" in s:
            return "Windows 10 Pro"
        elif "windows 10 home" in s:
            return "Windows 10 Home"
        elif "windows 10" in s or "win 10" in s:
            return "Windows 10"
        elif "windows 7" in s or "win 7" in s:
            return "Windows 7"
        elif "windows 8" in s:
            return "Windows 8"
        elif "windows" in s or "win " in s:
            return "Windows Other"
        elif "mac" in s:
            return "macOS"
        elif "chrome" in s:
            return "Chrome OS"
        elif "linux" in s:
            return "Linux"
        else:
            return "Other"

    # ─────────────────────────────────────────────────────────
    # تحميل ومعالجة البيانات
    # ─────────────────────────────────────────────────────────
    def load_and_prepare_data(self):
        """
        تحميل ملف CSV ومعالجة بيانات أمازون:
        - تنظيف الأسعار من رموز العملة والفواصل
        - استخراج سعة الرام والتخزين كأرقام
        - استخراج حجم الشاشة كرقم
        - تصنيف المعالج وكرت الشاشة ونظام التشغيل
        - حذف الصفوف التي تحتوي على قيم ناقصة جوهرية
        """
        try:
            # تحديد المسار المطلق لملف البيانات بناءً على موقع السكربت
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, self.DATA_PATH)

            self.df = pd.read_csv(csv_path, encoding="utf-8")
            df = self.df.copy()

            # ── تنظيف السعر (العمود الهدف) ──
            # "$1,599.00 " → 1599.00
            df["Price_USD"] = df["price"].apply(self._parse_price)

            # ── استخراج سعة الرام بالجيجابايت ──
            # "8 GB" → 8 ، "64 MB" → 0.064
            df["Ram_GB"] = df["ram"].apply(self._parse_ram)

            # ── استخراج سعة التخزين بالجيجابايت ──
            # "512 GB" → 512 ، "1 TB" → 1000
            df["Storage_GB"] = df["harddisk"].apply(self._parse_storage)

            # ── استخراج حجم الشاشة بالبوصة ──
            # "15.6 Inches" → 15.6
            df["Screen_Inches"] = df["screen_size"].apply(self._parse_screen_size)

            # ── استخراج سرعة المعالج ──
            # "2.4 GHz" → 2.4
            df["Cpu_Speed_GHz"] = df["cpu_speed"].apply(self._parse_cpu_speed)

            # ── تصنيف المعالج إلى فئات رئيسية ──
            df["Cpu_Category"] = df["cpu"].apply(self._categorize_cpu)

            # ── تصنيف كرت الشاشة إلى فئات رئيسية ──
            df["Graphics_Type"] = df["graphics"].apply(self._categorize_graphics)

            # ── تبسيط أنظمة التشغيل ──
            df["OS_Category"] = df["OS"].apply(self._categorize_os)

            # ── توحيد أسماء الماركات (بعضها مكتوب بأحرف كبيرة) ──
            brand_map = {
                "ASUS": "Asus", "DELL": "Dell", "LENOVO": "Lenovo",
                "MAINGEAR": "Maingear", "MICROTELLA": "Microtella",
            }
            df["brand"] = df["brand"].replace(brand_map)

            # ── حذف الصفوف ذات القيم الناقصة في الأعمدة الجوهرية ──
            essential_cols = ["Price_USD", "brand", "Ram_GB", "Storage_GB",
                              "Screen_Inches", "Cpu_Category", "Graphics_Type", "OS_Category"]
            df = df.dropna(subset=essential_cols)

            # ── حذف القيم غير المنطقية ──
            df = df[df["Price_USD"] > 0]
            df = df[df["Ram_GB"] > 0]
            df = df[df["Storage_GB"] > 0]
            df = df[df["Screen_Inches"] > 5]  # شاشات أقل من 5 بوصة غير منطقية للابتوب
            df = df[df["Screen_Inches"] < 25]  # شاشات أكبر من 25 بوصة غير منطقية

            # ── ملء سرعة المعالج الناقصة بالوسيط ──
            df["Cpu_Speed_GHz"] = df["Cpu_Speed_GHz"].fillna(df["Cpu_Speed_GHz"].median())

            # ── فلترة الماركات النادرة (أقل من 5 أجهزة) ──
            brand_counts = df["brand"].value_counts()
            rare_brands = brand_counts[brand_counts < 5].index
            df.loc[df["brand"].isin(rare_brands), "brand"] = "Other"

            self.df_clean = df.reset_index(drop=True)
            return True

        except FileNotFoundError:
            messagebox.showerror(
                "خطأ في الملف",
                f"لم يتم العثور على الملف:\n{self.DATA_PATH}\nتأكد من وجوده في نفس مجلد البرنامج."
            )
            return False
        except Exception as e:
            messagebox.showerror(
                "خطأ في معالجة البيانات",
                f"حدث خطأ أثناء معالجة البيانات:\n{e}"
            )
            return False

    # ─────────────────────────────────────────────────────────
    # بناء وتدريب النموذج
    # ─────────────────────────────────────────────────────────
    def build_and_train(self):
        """
        بناء خط أنابيب (Pipeline) يشمل:
        1. ColumnTransformer لمعالجة الميزات (OneHotEncoder للنصوص، StandardScaler للأرقام)
        2. RandomForestRegressor كنموذج التنبؤ
        ثم تدريبه وحساب مقاييس الأداء وحفظ النموذج
        """
        try:
            if self.df_clean is None:
                return False

            # تحديد مصفوفة الميزات (X) والهدف (y)
            all_features = self.cat_features + self.num_features
            X = self.df_clean[all_features]
            y = self.df_clean["Price_USD"]

            # تقسيم البيانات: 80% للتدريب و 20% للاختبار
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # ── بناء محول الأعمدة (ColumnTransformer) ──
            preprocessor = ColumnTransformer(
                transformers=[
                    # تشفير الأعمدة النصية باستخدام OneHotEncoder
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                     self.cat_features),
                    # توحيد الأعمدة الرقمية باستخدام StandardScaler
                    ("num", StandardScaler(), self.num_features),
                ],
                remainder="drop"  # تجاهل أي أعمدة غير محددة
            )

            # ── بناء خط الأنابيب الكامل ──
            self.pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(
                    n_estimators=300,      # عدد أشجار الغابة
                    max_depth=20,          # أقصى عمق للشجرة
                    min_samples_split=5,   # الحد الأدنى للتقسيم
                    random_state=42,
                    n_jobs=-1              # استخدام جميع الأنوية
                ))
            ])

            # ── تدريب النموذج ──
            self.pipeline.fit(X_train, y_train)

            # ── حساب مقاييس الأداء على بيانات الاختبار ──
            y_pred = self.pipeline.predict(X_test)
            self.metrics = {
                "r2":   round(r2_score(y_test, y_pred), 4),
                "mae":  round(mean_absolute_error(y_test, y_pred), 2),
                "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            }

            # ── حفظ النموذج المدرب ──
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, self.MODEL_PATH)
            joblib.dump({
                "pipeline": self.pipeline,
                "metrics":  self.metrics,
            }, model_path)

            self.is_trained = True
            return True

        except Exception as e:
            messagebox.showerror("خطأ في التدريب", f"حدث خطأ أثناء تدريب النموذج:\n{e}")
            return False

    # ─────────────────────────────────────────────────────────
    # تحميل النموذج المحفوظ مسبقاً
    # ─────────────────────────────────────────────────────────
    def load_saved_model(self):
        """
        محاولة تحميل نموذج مدرب سابقاً من ملف joblib.
        يُرجع True إذا تم التحميل بنجاح، False إذا لم يكن الملف موجوداً.
        """
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, self.MODEL_PATH)

            if os.path.exists(model_path):
                data = joblib.load(model_path)
                self.pipeline = data["pipeline"]
                self.metrics = data["metrics"]
                self.is_trained = True
                return True
            return False
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────
    # تنفيذ التنبؤ
    # ─────────────────────────────────────────────────────────
    def predict(self, input_dict):
        """
        تنفيذ تنبؤ بسعر اللابتوب بناءً على مدخلات المستخدم.
        input_dict: قاموس يحتوي على قيم الميزات
        يُرجع السعر المتوقع بالدولار
        """
        try:
            if not self.is_trained or self.pipeline is None:
                return None

            # بناء إطار بيانات من المدخلات
            all_features = self.cat_features + self.num_features
            input_df = pd.DataFrame([input_dict], columns=all_features)
            prediction = self.pipeline.predict(input_df)[0]
            return round(prediction, 2)

        except Exception as e:
            messagebox.showerror("خطأ في التنبؤ", f"حدث خطأ أثناء التنبؤ:\n{e}")
            return None

    # ─────────────────────────────────────────────────────────
    # استخراج القيم الفريدة للقوائم المنسدلة
    # ─────────────────────────────────────────────────────────
    def get_unique_values(self, column):
        """استخراج القيم الفريدة من عمود معين في البيانات المنظفة"""
        if self.df_clean is not None and column in self.df_clean.columns:
            return sorted(self.df_clean[column].dropna().unique().tolist())
        return []


# ══════════════════════════════════════════════════════════════
# الفئة الثانية: واجهة المستخدم الرسومية (Dashboard UI)
# ══════════════════════════════════════════════════════════════

class LaptopPriceDashboard(ctk.CTk):
    """
    فئة الواجهة الرسومية الرئيسية - لوحة التحكم:
    - قائمة جانبية بأزرار التحكم
    - منطقة محتوى رئيسية بأقسام الإدخال والنتائج والمقاييس
    - دعم تبديل المظهر (Dark/Light)
    - محاذاة من اليمين لليسار (RTL)
    """

    def __init__(self):
        super().__init__()

        # ── إعدادات النافذة الرئيسية ──
        self.title("🖥️  نظام التنبؤ بأسعار اللابتوب  |  Laptop Price Predictor")
        self.geometry("1280x780")
        self.minsize(1100, 700)

        # ── تحديد المظهر الافتراضي ──
        self.current_mode = "dark"
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.colors = DARK

        # ── تهيئة محرك تعلم الآلة ──
        self.model = LaptopPriceModel()

        # ── متغيرات الواجهة ──
        self.input_vars = {}       # متغيرات حقول الإدخال
        self.option_menus = {}     # القوائم المنسدلة
        self.entries = {}          # حقول الإدخال الرقمية

        # ── بناء الواجهة ──
        self._build_layout()

        # ── تحميل البيانات والنموذج في خيط منفصل (لعدم تجميد الواجهة) ──
        self._init_model_async()

    # ═════════════════════════════════════════════════════════
    # بناء التخطيط العام
    # ═════════════════════════════════════════════════════════
    def _build_layout(self):
        """بناء الهيكل العام: قائمة جانبية + منطقة محتوى رئيسية"""

        c = self.colors

        # ── إعداد الشبكة الرئيسية ──
        self.grid_columnconfigure(1, weight=1)  # المحتوى الرئيسي يتمدد
        self.grid_rowconfigure(0, weight=1)

        # ══════════════════════════════════════
        # القائمة الجانبية (Sidebar)
        # ══════════════════════════════════════
        self.sidebar = ctk.CTkFrame(
            self, width=260, corner_radius=0,
            fg_color=c["bg_sidebar"],
            border_width=0
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        self._build_sidebar()

        # ══════════════════════════════════════
        # المنطقة الرئيسية (Main Content)
        # ══════════════════════════════════════
        self.main_area = ctk.CTkScrollableFrame(
            self, corner_radius=0,
            fg_color=c["bg_primary"],
            scrollbar_button_color=c["accent"],
            scrollbar_button_hover_color=c["accent_hover"]
        )
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.main_area.grid_columnconfigure(0, weight=1)

        self._build_main_content()

    # ═════════════════════════════════════════════════════════
    # بناء القائمة الجانبية
    # ═════════════════════════════════════════════════════════
    def _build_sidebar(self):
        """بناء محتويات القائمة الجانبية: الشعار، الأزرار، والمعلومات"""
        c = self.colors

        # ── شعار وعنوان التطبيق ──
        logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_frame.pack(fill="x", padx=20, pady=(30, 5))

        ctk.CTkLabel(
            logo_frame,
            text="💻",
            font=ctk.CTkFont(size=48),
            text_color=c["accent"],
        ).pack(anchor="center")

        ctk.CTkLabel(
            logo_frame,
            text="Laptop Price",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color=c["text_primary"],
        ).pack(anchor="center", pady=(5, 0))

        ctk.CTkLabel(
            logo_frame,
            text="Predictor",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color=c["accent"],
        ).pack(anchor="center", pady=(0, 0))

        ctk.CTkLabel(
            logo_frame,
            text="نظام التنبؤ بالأسعار",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=c["text_secondary"],
        ).pack(anchor="center", pady=(4, 0))

        # ── خط فاصل ──
        ctk.CTkFrame(
            self.sidebar, height=2, fg_color=c["border"]
        ).pack(fill="x", padx=30, pady=(25, 25))

        # ── حالة النموذج ──
        self.model_status_frame = ctk.CTkFrame(
            self.sidebar, fg_color=c["bg_card"], corner_radius=12
        )
        self.model_status_frame.pack(fill="x", padx=20, pady=(0, 15))

        self.model_status_icon = ctk.CTkLabel(
            self.model_status_frame, text="⏳",
            font=ctk.CTkFont(size=18),
        )
        self.model_status_icon.pack(anchor="center", pady=(12, 2))

        self.model_status_label = ctk.CTkLabel(
            self.model_status_frame,
            text="جارٍ تحميل النموذج...",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=c["warning"],
        )
        self.model_status_label.pack(anchor="center", pady=(0, 12))

        # ── معلومات الداتا سيت ──
        self.dataset_info_label = ctk.CTkLabel(
            self.sidebar,
            text="📊 Amazon Laptop Prices",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=c["text_muted"],
        )
        self.dataset_info_label.pack(anchor="center", pady=(0, 10))

        # ── زر تبديل المظهر ──
        self.theme_btn = ctk.CTkButton(
            self.sidebar,
            text="☀️  الوضع الفاتح",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            fg_color=c["bg_card"],
            hover_color=c["border"],
            text_color=c["text_primary"],
            corner_radius=10,
            height=42,
            command=self._toggle_theme,
            anchor="center"
        )
        self.theme_btn.pack(fill="x", padx=20, pady=(5, 8))

        # ── زر إعادة الضبط ──
        self.reset_btn = ctk.CTkButton(
            self.sidebar,
            text="🔄  إعادة ضبط المدخلات",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            fg_color=c["bg_card"],
            hover_color=c["border"],
            text_color=c["text_primary"],
            corner_radius=10,
            height=42,
            command=self._reset_inputs,
            anchor="center"
        )
        self.reset_btn.pack(fill="x", padx=20, pady=(0, 8))

        # ── زر إعادة تدريب النموذج ──
        self.retrain_btn = ctk.CTkButton(
            self.sidebar,
            text="🧠  إعادة تدريب النموذج",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            fg_color=c["bg_card"],
            hover_color=c["border"],
            text_color=c["warning"],
            corner_radius=10,
            height=42,
            command=self._retrain_model,
            anchor="center"
        )
        self.retrain_btn.pack(fill="x", padx=20, pady=(0, 8))

        # ── ملء الفراغ السفلي ──
        spacer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        spacer.pack(fill="both", expand=True)

        # ── معلومات المطور ──
        ctk.CTkLabel(
            self.sidebar,
            text="Powered by Abdullah & Osama",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=c["text_muted"],
        ).pack(anchor="center", pady=(0, 5))

        ctk.CTkLabel(
            self.sidebar,
            text="v2.0  |  Amazon Dataset",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=c["text_muted"],
        ).pack(anchor="center", pady=(0, 20))

    # ═════════════════════════════════════════════════════════
    # بناء المحتوى الرئيسي
    # ═════════════════════════════════════════════════════════
    def _build_main_content(self):
        """بناء أقسام المحتوى: العنوان، المدخلات، زر التنبؤ، النتيجة، المقاييس"""
        c = self.colors

        # ══════════════════════════════════════
        # 1. شريط العنوان العلوي
        # ══════════════════════════════════════
        header = ctk.CTkFrame(self.main_area, fg_color="transparent")
        header.pack(fill="x", padx=30, pady=(25, 5))

        ctk.CTkLabel(
            header,
            text="🔮  لوحة التنبؤ بالأسعار",
            font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"),
            text_color=c["text_primary"],
            anchor="e"
        ).pack(side="right")

        ctk.CTkLabel(
            header,
            text="أدخل مواصفات اللابتوب للحصول على السعر المتوقع بالدولار",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=c["text_secondary"],
            anchor="w"
        ).pack(side="left", pady=(6, 0))

        # ══════════════════════════════════════
        # 2. قسم المدخلات (Input Section)
        # ══════════════════════════════════════
        input_container = ctk.CTkFrame(
            self.main_area, fg_color=c["bg_card"],
            corner_radius=16, border_width=1, border_color=c["border"]
        )
        input_container.pack(fill="x", padx=25, pady=(15, 10))

        # عنوان القسم
        section_header = ctk.CTkFrame(input_container, fg_color="transparent")
        section_header.pack(fill="x", padx=25, pady=(18, 10))

        ctk.CTkLabel(
            section_header,
            text="📋  مواصفات الجهاز",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=c["text_primary"],
            anchor="e"
        ).pack(side="right")

        # ── شبكة المدخلات (عمودين) ──
        grid_frame = ctk.CTkFrame(input_container, fg_color="transparent")
        grid_frame.pack(fill="x", padx=20, pady=(5, 20))
        grid_frame.grid_columnconfigure(0, weight=1)
        grid_frame.grid_columnconfigure(1, weight=1)

        # تعريف حقول الإدخال الخاصة بداتا سيت أمازون
        # (المفتاح، العنوان، النوع، العمود، الصف)
        fields = [
            # ── العمود الأيمن (Column 1) - القوائم المنسدلة ──
            ("brand",          "🏢  الشركة المصنعة",       "option", 1, 0),
            ("Cpu_Category",   "⚡  فئة المعالج",           "option", 1, 1),
            ("Graphics_Type",  "🎮  نوع كرت الشاشة",      "option", 1, 2),
            ("OS_Category",    "🖥️  نظام التشغيل",        "option", 1, 3),
            # ── العمود الأيسر (Column 0) - الحقول الرقمية ──
            ("Screen_Inches",  "📏  حجم الشاشة (بوصة)",   "entry",  0, 0),
            ("Ram_GB",         "🧠  سعة الرام (GB)",       "entry",  0, 1),
            ("Storage_GB",     "💾  سعة التخزين (GB)",     "entry",  0, 2),
            ("Cpu_Speed_GHz",  "⚡  سرعة المعالج (GHz)",   "entry",  0, 3),
        ]

        for key, label, field_type, col, row in fields:
            self._create_input_field(grid_frame, key, label, field_type, col, row, c)

        # ══════════════════════════════════════
        # 3. زر التنبؤ
        # ══════════════════════════════════════
        self.predict_btn = ctk.CTkButton(
            self.main_area,
            text="🚀  تنبؤ بالسعر الآن",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            fg_color=c["accent"],
            hover_color=c["accent_hover"],
            text_color="#ffffff",
            corner_radius=14,
            height=55,
            command=self._on_predict,
        )
        self.predict_btn.pack(fill="x", padx=25, pady=(10, 10))

        # ══════════════════════════════════════
        # 4. بطاقة النتيجة (Result Card)
        # ══════════════════════════════════════
        self.result_card = ctk.CTkFrame(
            self.main_area, fg_color=c["result_bg"],
            corner_radius=16, border_width=2, border_color=c["accent"],
            height=130
        )
        self.result_card.pack(fill="x", padx=25, pady=(5, 10))
        self.result_card.pack_propagate(False)

        ctk.CTkLabel(
            self.result_card,
            text="💰  السعر المتوقع",
            font=ctk.CTkFont(family="Segoe UI", size=15),
            text_color=c["text_secondary"],
        ).pack(anchor="center", pady=(18, 0))

        self.price_label = ctk.CTkLabel(
            self.result_card,
            text="—  $",
            font=ctk.CTkFont(family="Consolas", size=44, weight="bold"),
            text_color=c["success"],
        )
        self.price_label.pack(anchor="center", pady=(2, 0))

        self.price_note = ctk.CTkLabel(
            self.result_card,
            text="أدخل المواصفات ثم اضغط على زر التنبؤ",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=c["text_muted"],
        )
        self.price_note.pack(anchor="center", pady=(0, 10))

        # ══════════════════════════════════════
        # 5. قسم مقاييس الأداء (Metrics)
        # ══════════════════════════════════════
        metrics_header = ctk.CTkFrame(self.main_area, fg_color="transparent")
        metrics_header.pack(fill="x", padx=30, pady=(15, 5))
        ctk.CTkLabel(
            metrics_header,
            text="📊  مقاييس أداء النموذج",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=c["text_primary"],
            anchor="e"
        ).pack(side="right")

        metrics_row = ctk.CTkFrame(self.main_area, fg_color="transparent")
        metrics_row.pack(fill="x", padx=25, pady=(5, 25))
        metrics_row.grid_columnconfigure(0, weight=1)
        metrics_row.grid_columnconfigure(1, weight=1)
        metrics_row.grid_columnconfigure(2, weight=1)

        # بطاقة R² Score
        self.r2_card = self._create_metric_card(
            metrics_row, "R² Score", "معامل التحديد", "—", c["success"], 0
        )
        # بطاقة MAE
        self.mae_card = self._create_metric_card(
            metrics_row, "MAE", "متوسط الخطأ المطلق", "—", c["warning"], 1
        )
        # بطاقة RMSE
        self.rmse_card = self._create_metric_card(
            metrics_row, "RMSE", "جذر متوسط مربع الخطأ", "—", c["danger"], 2
        )

    # ─────────────────────────────────────────────────────────
    # إنشاء حقل إدخال واحد (قائمة منسدلة أو حقل رقمي)
    # ─────────────────────────────────────────────────────────
    def _create_input_field(self, parent, key, label_text, field_type, col, row, c):
        """
        إنشاء حقل إدخال واحد داخل الشبكة:
        - key: مفتاح الحقل (اسم الميزة)
        - label_text: النص المعروض
        - field_type: "option" أو "entry"
        - col, row: موقع الحقل في الشبكة
        """
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=col, sticky="ew", padx=10, pady=6)
        frame.grid_columnconfigure(0, weight=1)

        # عنوان الحقل (محاذاة لليمين - RTL)
        ctk.CTkLabel(
            frame, text=label_text,
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=c["text_secondary"],
            anchor="e"
        ).grid(row=0, column=0, sticky="e", padx=(0, 5), pady=(0, 3))

        if field_type == "option":
            # ── قائمة منسدلة ──
            var = ctk.StringVar(value="— اختر —")
            self.input_vars[key] = var

            # القيم الافتراضية (سيتم تحديثها بعد تحميل البيانات)
            values = ["جارٍ التحميل..."]

            menu = ctk.CTkOptionMenu(
                frame, variable=var, values=values,
                font=ctk.CTkFont(family="Segoe UI", size=13),
                dropdown_font=ctk.CTkFont(family="Segoe UI", size=13),
                fg_color=c["bg_input"],
                button_color=c["accent"],
                button_hover_color=c["accent_hover"],
                dropdown_fg_color=c["bg_card"],
                dropdown_hover_color=c["accent"],
                dropdown_text_color=c["text_primary"],
                text_color=c["text_primary"],
                corner_radius=8,
                height=38,
                anchor="e"
            )
            menu.grid(row=1, column=0, sticky="ew", padx=(0, 5))
            self.option_menus[key] = menu

        else:
            # ── حقل إدخال رقمي ──
            var = ctk.StringVar(value="")
            self.input_vars[key] = var

            # تسجيل دالة التحقق للسماح بالأرقام فقط
            vcmd = (self.register(self._validate_numeric_input), "%P")

            entry = ctk.CTkEntry(
                frame, textvariable=var,
                font=ctk.CTkFont(family="Consolas", size=14),
                fg_color=c["bg_input"],
                text_color=c["text_primary"],
                border_color=c["border"],
                border_width=1,
                corner_radius=8,
                height=38,
                justify="right",  # محاذاة النص لليمين
                placeholder_text="0",
                validate="key",
                validatecommand=vcmd,
            )
            entry.grid(row=1, column=0, sticky="ew", padx=(0, 5))
            self.entries[key] = entry

    # ─────────────────────────────────────────────────────────
    # إنشاء بطاقة مقياس أداء
    # ─────────────────────────────────────────────────────────
    def _create_metric_card(self, parent, title, subtitle, value, color, col):
        """
        إنشاء بطاقة مقياس أداء واحدة (R², MAE, RMSE)
        تُرجع القاموس الذي يحتوي على عناصر البطاقة للتحديث لاحقاً
        """
        c = self.colors

        card = ctk.CTkFrame(
            parent, fg_color=c["metric_bg"],
            corner_radius=14, border_width=1, border_color=c["border"]
        )
        card.grid(row=0, column=col, sticky="ew", padx=8, pady=5)

        ctk.CTkLabel(
            card, text=title,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=c["text_secondary"],
        ).pack(anchor="center", pady=(15, 0))

        value_label = ctk.CTkLabel(
            card, text=value,
            font=ctk.CTkFont(family="Consolas", size=28, weight="bold"),
            text_color=color,
        )
        value_label.pack(anchor="center", pady=(4, 0))

        ctk.CTkLabel(
            card, text=subtitle,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=c["text_muted"],
        ).pack(anchor="center", pady=(2, 15))

        return {"frame": card, "value_label": value_label}

    # ═════════════════════════════════════════════════════════
    # التحقق من صحة المدخلات الرقمية
    # ═════════════════════════════════════════════════════════
    def _validate_numeric_input(self, value):
        """
        دالة تحقق لحقول الإدخال الرقمية:
        تسمح فقط بالأرقام والنقطة العشرية والحقل الفارغ
        """
        if value == "":
            return True
        try:
            float(value)
            return True
        except ValueError:
            return False

    # ═════════════════════════════════════════════════════════
    # تهيئة النموذج بشكل غير متزامن
    # ═════════════════════════════════════════════════════════
    def _init_model_async(self):
        """تحميل البيانات والنموذج في خيط منفصل لعدم تجميد الواجهة"""

        def _task():
            # الخطوة 1: تحميل ومعالجة البيانات
            success = self.model.load_and_prepare_data()
            if not success:
                self.after(0, lambda: self._update_model_status("error"))
                return

            # الخطوة 2: تحديث القوائم المنسدلة بالقيم من البيانات
            self.after(0, self._populate_dropdowns)

            # الخطوة 3: محاولة تحميل النموذج المحفوظ
            loaded = self.model.load_saved_model()
            if not loaded:
                # النموذج غير موجود → تدريب جديد
                self.after(0, lambda: self._update_model_status("training"))
                success = self.model.build_and_train()
                if not success:
                    self.after(0, lambda: self._update_model_status("error"))
                    return

            # الخطوة 4: تحديث الواجهة بمقاييس الأداء
            self.after(0, self._update_metrics_display)
            self.after(0, lambda: self._update_model_status("ready"))

        thread = threading.Thread(target=_task, daemon=True)
        thread.start()

    # ─────────────────────────────────────────────────────────
    # تحديث حالة النموذج في القائمة الجانبية
    # ─────────────────────────────────────────────────────────
    def _update_model_status(self, status):
        """تحديث مؤشر حالة النموذج في القائمة الجانبية"""
        c = self.colors
        if status == "ready":
            self.model_status_icon.configure(text="✅")
            self.model_status_label.configure(
                text="النموذج جاهز للعمل",
                text_color=c["success"]
            )
            # عرض عدد الأجهزة في الداتا سيت
            if self.model.df_clean is not None:
                count = len(self.model.df_clean)
                self.dataset_info_label.configure(
                    text=f"📊 {count} جهاز في الداتا سيت"
                )
        elif status == "training":
            self.model_status_icon.configure(text="🧠")
            self.model_status_label.configure(
                text="جارٍ تدريب النموذج...",
                text_color=c["warning"]
            )
        elif status == "error":
            self.model_status_icon.configure(text="❌")
            self.model_status_label.configure(
                text="خطأ في النموذج!",
                text_color=c["danger"]
            )

    # ─────────────────────────────────────────────────────────
    # تحديث القوائم المنسدلة بالقيم الفعلية من البيانات
    # ─────────────────────────────────────────────────────────
    def _populate_dropdowns(self):
        """تعبئة القوائم المنسدلة بالقيم الفريدة من ملف البيانات المنظف"""

        # ربط كل مفتاح بالعمود المناسب في البيانات المنظفة
        dropdown_columns = {
            "brand":          "brand",
            "Cpu_Category":   "Cpu_Category",
            "Graphics_Type":  "Graphics_Type",
            "OS_Category":    "OS_Category",
        }

        for key, column in dropdown_columns.items():
            if key in self.option_menus:
                values = self.model.get_unique_values(column)
                if values:
                    str_values = [str(v) for v in values]
                    self.option_menus[key].configure(values=str_values)
                    self.input_vars[key].set(str_values[0])

    # ─────────────────────────────────────────────────────────
    # تحديث عرض مقاييس الأداء
    # ─────────────────────────────────────────────────────────
    def _update_metrics_display(self):
        """تحديث بطاقات مقاييس الأداء بالقيم الفعلية من النموذج"""
        m = self.model.metrics
        if m:
            self.r2_card["value_label"].configure(text=f"{m.get('r2', '—')}")
            self.mae_card["value_label"].configure(text=f"${m.get('mae', '—')}")
            self.rmse_card["value_label"].configure(text=f"${m.get('rmse', '—')}")

    # ═════════════════════════════════════════════════════════
    # معالجة حدث التنبؤ
    # ═════════════════════════════════════════════════════════
    def _on_predict(self):
        """
        معالجة ضغط زر التنبؤ:
        1. التحقق من جاهزية النموذج
        2. التحقق من صحة المدخلات
        3. تنفيذ التنبؤ وعرض النتيجة
        """
        c = self.colors

        # ── التحقق من جاهزية النموذج ──
        if not self.model.is_trained:
            messagebox.showwarning(
                "النموذج غير جاهز",
                "النموذج لم يتم تحميله بعد.\nيرجى الانتظار حتى اكتمال التدريب."
            )
            return

        # ── تغيير حالة الزر (تغذية راجعة) ──
        self.predict_btn.configure(
            text="⏳  جارٍ التنبؤ...",
            state="disabled",
            fg_color=c["text_muted"]
        )
        self.update_idletasks()

        try:
            # ── جمع المدخلات والتحقق منها ──
            input_dict = {}

            # الحقول النصية (القوائم المنسدلة)
            cat_labels = {
                "brand": "الشركة المصنعة",
                "Cpu_Category": "فئة المعالج",
                "Graphics_Type": "نوع كرت الشاشة",
                "OS_Category": "نظام التشغيل",
            }
            for key, arabic_name in cat_labels.items():
                val = self.input_vars[key].get()
                if val in ("— اختر —", "جارٍ التحميل..."):
                    messagebox.showwarning(
                        "حقل مطلوب",
                        f"يرجى اختيار قيمة لحقل: {arabic_name}"
                    )
                    self._restore_predict_btn()
                    return
                input_dict[key] = val

            # الحقول الرقمية
            numeric_fields = {
                "Screen_Inches":  "حجم الشاشة",
                "Ram_GB":         "سعة الرام",
                "Storage_GB":     "سعة التخزين",
                "Cpu_Speed_GHz":  "سرعة المعالج",
            }

            for key, arabic_name in numeric_fields.items():
                val_str = self.input_vars[key].get().strip()
                if val_str == "":
                    messagebox.showwarning(
                        "حقل فارغ",
                        f"يرجى إدخال قيمة لحقل: {arabic_name}"
                    )
                    self._restore_predict_btn()
                    return
                try:
                    val = float(val_str)
                except ValueError:
                    messagebox.showerror(
                        "قيمة غير صالحة",
                        f"القيمة المدخلة في حقل «{arabic_name}» ليست رقماً صحيحاً."
                    )
                    self._restore_predict_btn()
                    return

                if val < 0:
                    messagebox.showerror(
                        "قيمة سالبة",
                        f"لا يمكن إدخال قيمة سالبة في حقل «{arabic_name}»."
                    )
                    self._restore_predict_btn()
                    return
                input_dict[key] = val

            # ── تنفيذ التنبؤ ──
            prediction = self.model.predict(input_dict)

            if prediction is not None:
                # عرض النتيجة بتنسيق جميل (بالدولار)
                formatted = f"${prediction:,.2f}"
                self.price_label.configure(text=formatted)
                self.price_note.configure(
                    text="✅  تم التنبؤ بنجاح بناءً على المواصفات المدخلة",
                    text_color=c["success"]
                )
            else:
                self.price_label.configure(text="خطأ!")
                self.price_note.configure(
                    text="❌  حدث خطأ أثناء التنبؤ، يرجى المحاولة مرة أخرى",
                    text_color=c["danger"]
                )

        except Exception as e:
            messagebox.showerror("خطأ غير متوقع", f"حدث خطأ غير متوقع:\n{e}")

        finally:
            self._restore_predict_btn()

    # ─────────────────────────────────────────────────────────
    # استعادة حالة زر التنبؤ
    # ─────────────────────────────────────────────────────────
    def _restore_predict_btn(self):
        """إعادة زر التنبؤ لحالته الطبيعية بعد انتهاء العملية"""
        c = self.colors
        self.predict_btn.configure(
            text="🚀  تنبؤ بالسعر الآن",
            state="normal",
            fg_color=c["accent"]
        )

    # ═════════════════════════════════════════════════════════
    # إعادة ضبط المدخلات
    # ═════════════════════════════════════════════════════════
    def _reset_inputs(self):
        """إعادة جميع حقول الإدخال والقوائم المنسدلة لقيمها الافتراضية"""
        c = self.colors

        # إعادة ضبط الحقول الرقمية
        for key in self.entries:
            self.input_vars[key].set("")

        # إعادة ضبط القوائم المنسدلة
        for key in self.option_menus:
            menu = self.option_menus[key]
            try:
                values = menu.cget("values")
                if values:
                    self.input_vars[key].set(values[0])
                else:
                    self.input_vars[key].set("— اختر —")
            except Exception:
                self.input_vars[key].set("— اختر —")

        # إعادة ضبط بطاقة النتيجة
        self.price_label.configure(text="—  $")
        self.price_note.configure(
            text="أدخل المواصفات ثم اضغط على زر التنبؤ",
            text_color=c["text_muted"]
        )

    # ═════════════════════════════════════════════════════════
    # إعادة تدريب النموذج
    # ═════════════════════════════════════════════════════════
    def _retrain_model(self):
        """إعادة تدريب النموذج من الصفر (حذف النموذج المحفوظ وإعادة التدريب)"""

        confirm = messagebox.askyesno(
            "تأكيد إعادة التدريب",
            "هل تريد إعادة تدريب النموذج من الصفر؟\nقد تستغرق هذه العملية بضع ثوانٍ."
        )
        if not confirm:
            return

        self._update_model_status("training")
        self.retrain_btn.configure(state="disabled", text="⏳  جارٍ التدريب...")

        def _task():
            try:
                # حذف النموذج المحفوظ إن وجد
                base_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(base_dir, self.model.MODEL_PATH)
                if os.path.exists(model_path):
                    os.remove(model_path)

                # إعادة التدريب
                success = self.model.build_and_train()
                if success:
                    self.after(0, self._update_metrics_display)
                    self.after(0, lambda: self._update_model_status("ready"))
                else:
                    self.after(0, lambda: self._update_model_status("error"))

            except Exception as e:
                self.after(0, lambda: messagebox.showerror("خطأ", f"خطأ في إعادة التدريب:\n{e}"))
                self.after(0, lambda: self._update_model_status("error"))
            finally:
                self.after(0, lambda: self.retrain_btn.configure(
                    state="normal", text="🧠  إعادة تدريب النموذج"
                ))

        thread = threading.Thread(target=_task, daemon=True)
        thread.start()

    # ═════════════════════════════════════════════════════════
    # تبديل المظهر (Dark / Light)
    # ═════════════════════════════════════════════════════════
    def _toggle_theme(self):
        """
        تبديل بين الوضع الداكن والفاتح:
        يُعيد بناء الواجهة بالكامل بالألوان الجديدة
        """
        if self.current_mode == "dark":
            self.current_mode = "light"
            self.colors = LIGHT
            ctk.set_appearance_mode("light")
        else:
            self.current_mode = "dark"
            self.colors = DARK
            ctk.set_appearance_mode("dark")

        # حفظ القيم الحالية قبل إعادة البناء
        saved_values = {}
        for key, var in self.input_vars.items():
            saved_values[key] = var.get()

        saved_price = self.price_label.cget("text")
        saved_note = self.price_note.cget("text")

        # تدمير العناصر وإعادة بنائها
        for widget in self.winfo_children():
            widget.destroy()

        self.input_vars = {}
        self.option_menus = {}
        self.entries = {}

        self._build_layout()

        # استعادة القيم المحفوظة
        self.after(100, lambda: self._restore_values(saved_values, saved_price, saved_note))

    def _restore_values(self, saved_values, saved_price, saved_note):
        """استعادة قيم المدخلات بعد إعادة بناء الواجهة"""

        # إعادة تعبئة القوائم المنسدلة من البيانات
        if self.model.df_clean is not None:
            self._populate_dropdowns()

        # استعادة القيم المحفوظة
        for key, val in saved_values.items():
            if key in self.input_vars:
                self.input_vars[key].set(val)

        # استعادة النتيجة
        self.price_label.configure(text=saved_price)
        self.price_note.configure(text=saved_note)

        # استعادة المقاييس
        self._update_metrics_display()

        # تحديث زر المظهر
        if self.current_mode == "dark":
            self.theme_btn.configure(text="☀️  الوضع الفاتح")
        else:
            self.theme_btn.configure(text="🌙  الوضع الداكن")

        # تحديث حالة النموذج
        if self.model.is_trained:
            self._update_model_status("ready")


# ══════════════════════════════════════════════════════════════
# نقطة الدخول الرئيسية
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        app = LaptopPriceDashboard()
        app.mainloop()
    except Exception as e:
        messagebox.showerror(
            "خطأ فادح",
            f"حدث خطأ فادح أثناء تشغيل التطبيق:\n{e}"
        )
        sys.exit(1)
