import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.target_col = "Sleep Disorder"
        self.categorical_cols = ['Gender', 'Occupation', 'BMI Category']
        self.label_encoders = {}

    def load_data(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} tidak ditemukan.")
        self.data = pd.read_csv(filepath)
        print(f"[INFO] Data dimuat. Shape: {self.data.shape}")
        return self.data

    def clean_data(self):
        initial_count = len(self.data)
        self.data = self.data.drop_duplicates()
        print(f"[STEP 1] Duplikat dihapus: {initial_count - len(self.data)}")

        self.data = self.data.fillna("None")
        return self.data

    def encode_data(self):
        for col in self.categorical_cols:
            if col in self.data.columns:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
                print(f"[STEP 2] Encoding fitur: {col}")

        le_target = LabelEncoder()
        self.data[self.target_col] = le_target.fit_transform(self.data[self.target_col].astype(str))
        self.label_encoders['target'] = le_target
        print(f"[STEP 2] Encoding target: {self.target_col}")

        # Drop kolom yang tidak relevan
        if 'Person ID' in self.data.columns:
            self.data = self.data.drop('Person ID', axis=1)

        if 'Blood Pressure' in self.data.columns:
            self.data = self.data.drop('Blood Pressure', axis=1)

        return self.data

    def save_results(self, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.data.to_csv(output_path, index=False)

        # Simpan encoder di folder yang sama dengan output (biar path aman lokal & actions)
        encoder_path = os.path.join(os.path.dirname(output_path), "label_encoders.pkl")
        with open(encoder_path, "wb") as f:
            pickle.dump(self.label_encoders, f)

        print(f"[INFO] Data bersih disimpan ke: {output_path}")
        print(f"[INFO] Label encoders disimpan ke: {encoder_path}")

    def run_preprocessing_pipeline(self, input_path, output_path):
        print("--- MEMULAI AUTOMATE PREPROCESSING PIPELINE ---")
        self.load_data(input_path)
        self.clean_data()
        self.encode_data()
        self.save_results(output_path)
        print("--- SELESAI ---")

    # Alias biar tidak error kalau pemanggilan pakai run_pipeline()
    def run_pipeline(self, input_path, output_path):
        return self.run_preprocessing_pipeline(input_path, output_path)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Cari CSV di folder yang sama dengan script, kalau tidak ada cek 1 folder di atasnya
    candidate_1 = os.path.join(BASE_DIR, "Sleep_health_and_lifestyle_dataset.csv")
    candidate_2 = os.path.join(BASE_DIR, "..", "Sleep_health_and_lifestyle_dataset.csv")

    if os.path.exists(candidate_1):
        INPUT = candidate_1
    elif os.path.exists(candidate_2):
        INPUT = os.path.abspath(candidate_2)
    else:
        raise FileNotFoundError(
            "CSV tidak ditemukan.\n"
            "Pastikan file 'Sleep_health_and_lifestyle_dataset.csv' berada:\n"
            f"1) {candidate_1}\n"
            f"atau\n"
            f"2) {os.path.abspath(candidate_2)}"
        )

    OUTPUT = os.path.join(BASE_DIR, "namadataset_preprocessing", "data_bersih_eksperimen.csv")

    preprocessor = DataPreprocessor()
    preprocessor.run_pipeline(INPUT, OUTPUT)
