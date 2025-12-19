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
        
        # Simpan encoder untuk kebutuhan deployment nanti
        with open("preprocessing/label_encoders.pkl", "wb") as f:
            pickle.dump(self.label_encoders, f)
        print(f"[INFO] Data bersih disimpan ke: {output_path}")

    def run_pipeline(self, input_path, output_path):
        print("--- MEMULAI AUTOMATE PREPROCESSING PIPELINE ---")
        self.load_data(input_path)
        self.clean_data()
        self.encode_data()
        self.save_results(output_path)
        print("--- SELESAI ---")

if __name__ == "__main__":
    INPUT = "Sleep_health_and_lifestyle_dataset.csv"
    OUTPUT = "preprocessing/namadataset_preprocessing/data_bersih_eksperimen.csv"
    
    preprocessor = DataPreprocessor()
    preprocessor.run_pipeline(INPUT, OUTPUT)