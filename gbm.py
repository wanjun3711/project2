import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler
import streamlit as st
import joblib

# 读取训练集数据
train_data = pd.read_csv('train_data - 副本.csv')

# 分离输入特征和目标变量
X = train_data[['Age', 'Primary Site', 'Histologic', 'Tumor grade',
                   'T stage', 'N stage', 'Bone metastasis', 'Lung metastasis']]
y = train_data['Liver metastasis']

# 创建并训练GBM模型
gbm_model = GradientBoostingClassifier()
gbm_model.fit(X, y)

# 保存模型
model_path = 'gbm_model.pkl'
joblib.dump(gbm_model, model_path)

# 特征映射
class_mapping = {0: "No liver metastasis", 1: "Esophagus cancer liver metastasis"}
age_mapper = {"<70": 3, "70-80": 1, ">=80": 2}
primary_site_mapper = {"Upper third of esophagus": 4,"Middle third of esophagus": 1,
    "Lower third of esophagus": 2, "Overlapping lesion of esophagus": 3}
histologic_mapper = {"Adenocarcinoma": 2, "Squamous–cell carcinoma": 1}
tumor_grade_mapper = {"Grade I": 3, "Grade II": 1, "Grade III": 2}
t_stage_mapper = {"T1": 4, "T2": 1, "T3": 2, "T4": 3}
n_stage_mapper = {"N0": 4, "N1": 1, "N2": 2, "N3": 3}
chemotherapy_mapper = {"NO": 2, "Yes": 1}
bone_metastasis_mapper = {"NO": 2, "Yes": 1}
lung_metastasis_mapper = {"NO": 2, "Yes": 1}

# 预测函数
def predict_liver_metastasis(age, primary_site, histologic, tumor_grade,
                             t_stage, n_stage, chemotherapy, bone_metastasis, lung_metastasis):
    input_data = pd.DataFrame({
        'Age': [age_mapper[age]],
        'Primary Site': [primary_site_mapper[primary_site]],
        'Histologic': [histologic_mapper[histologic]],
        'Tumor grade': [tumor_grade_mapper[tumor_grade]],
        'T stage': [t_stage_mapper[t_stage]],
        'N stage': [n_stage_mapper[n_stage]],
        'Chemotherapy': [chemotherapy_mapper[chemotherapy]],
        'Bone metastasis': [bone_metastasis_mapper[bone_metastasis]],
        'Lung metastasis': [lung_metastasis_mapper[lung_metastasis]]
    })
    prediction = gbm_model.predict(input_data)[0]
    probability = gbm_model.predict_proba(input_data)[0][1]  # 获取属于类别1的概率
    class_label = class_mapping[prediction]
    return class_label, probability
# 创建Web应用程序
st.title("GBM Model Predicting Liver Metastasis of Esophageal Cancer")
st.sidebar.write("Variables")

age = st.sidebar.selectbox("Age", options=list(age_mapper.keys()))
primary_site = st.sidebar.selectbox("Primary site", options=list(primary_site_mapper.keys()))
histologic = st.sidebar.selectbox("Histologic", options=list(histologic_mapper.keys()))
tumor_grade = st.sidebar.selectbox("Tumor grade", options=list(tumor_grade_mapper.keys()))
t_stage = st.sidebar.selectbox("T stage", options=list(t_stage_mapper.keys()))
n_stage = st.sidebar.selectbox("N stage", options=list(n_stage_mapper.keys()))
chemotherapy = st.sidebar.selectbox("Chemotherapy", options=list(chemotherapy_mapper.keys()))
bone_metastasis = st.sidebar.selectbox("Bone metastasis", options=list(bone_metastasis_mapper.keys()))
lung_metastasis = st.sidebar.selectbox("Lung metastasis", options=list(lung_metastasis_mapper.keys()))

if st.button("Predict"):
    prediction, probability = predict_liver_metastasis(age, primary_site, histologic, tumor_grade,
                             t_stage, n_stage, surgery, radiation, chemotherapy, bone_metastasis, lung_metastasis)

    st.write("Probability of developing liver metastasis:", prediction)
    st.write("Probability of developing liver metastasis:", probability)
