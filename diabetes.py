import streamlit as st
import pandas as pd
import joblib

# 모델 & 스케일러 불러오기
log_model_eng = joblib.load("diabetes.pkl")
scaler = joblib.load("diabetes.scaler.pkl")

st.title("당뇨 예측 프로그램")

# 입력값
preg = st.number_input("임신횟수", min_value=0, step=1)
glucose = st.number_input("혈당")
bp = st.number_input("혈압")
skin = st.number_input("피부두께")
insulin = st.number_input("인슐린")
bmi = st.number_input("체질량지수(BMI)")
pedigree = st.number_input("당뇨내력가중치")
age = st.number_input("나이", min_value=0, step=1)

# 버튼
if st.button("예측하기"):

    # 입력 데이터
    input_data = pd.DataFrame(
        [[preg, glucose, bp, skin, insulin, bmi, pedigree, age]],
        columns=[
            '임신횟수', '혈당', '혈압', '피부두께',
            '인슐린', '체질량지수', '당뇨내력가중치', '나이'
        ]
    )

    # 파생 변수
    input_data['비만위험'] = (input_data['체질량지수'] >= 30).astype(int)
    input_data['고혈당'] = (input_data['인슐린'] >= 140).astype(int)
    input_data['고령'] = (input_data['나이'] >= 55).astype(int)
    input_data['대사위험'] = (
        (input_data['체질량지수'] >= 25).astype(int)
        + (input_data['인슐린'] >= 130).astype(int)
    )

    # 스케일링
    input_scaled = scaler.transform(input_data)

    # 예측
    predicted = log_model_eng.predict(input_scaled)
    prob = log_model_eng.predict_proba(input_scaled)

    # 결과 출력
    st.success(f"예측 결과: {'당뇨' if predicted[0] == 1 else '정상'}")
    st.write(f"당뇨 확률: {prob[0][1] * 100:.1f}%")