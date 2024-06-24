# 导入所需库
import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# 确保数据加载和模型评估函数可以被Streamlit调用
@st.cache(persist=True)
def load_data():
    file_path = r'C:\Users\田六六\Desktop\heroes_data.csv' 
    return pd.read_csv(file_path)

# 选择特征和目标变量
def select_features(data):
    X = data[['survival_ability', 'attack_damage', 'skill_effect', 'difficulty']]
    y = data['role']
    return X, y

# 划分训练集和测试集
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# 数据标准化
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

# 创建并评估逻辑回归模型
def evaluate_logistic_regression(X_train, y_train, X_test, y_test):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    logreg_pred = logreg.predict(X_test)
    accuracy = accuracy_score(y_test, logreg_pred)
    print(f"Logistic Regression Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, logreg_pred))

# 创建并评估随机森林模型
def evaluate_random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, rf_pred))

# 创建推荐函数
def recommend_heroes(data, user_preferences):
    recommended = data[(data['survival_ability'] >= user_preferences[0]) &
                       (data['attack_damage'] >= user_preferences[1]) &
                       (data['skill_effect'] >= user_preferences[2]) &
                       (data['difficulty'] <= user_preferences[3]) &
                       (data['role'].isin(['坦克', '法师', '战士', '刺客', '射手', '辅助']))]
    return recommended

# Streamlit应用逻辑
def main():
    st.title('王者荣耀英雄推荐系统')
    st.write('请根据您的偏好选择英雄：')

    # 用户偏好输入
    survival = st.slider('生存能力', 0, 10, 5)
    attack = st.slider('攻击伤害', 0, 10, 5)
    skill = st.slider('技能效果', 0, 10, 5)
    difficulty = st.slider('上手难度', 0, 10, 5)
    user_preferences = [survival, attack, skill, difficulty]

    # 加载数据
    data = load_data()

    # 选择特征和目标变量
    X, y = select_features(data)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 数据标准化
    X_train, X_test = scale_data(X_train, X_test)

    # 评估模型
    evaluate_logistic_regression(X_train, y_train, X_test, y_test)
    evaluate_random_forest(X_train, y_train, X_test, y_test)

    # 推荐英雄及其分路
    recommended_heroes_df = recommend_heroes(data, user_preferences)

    # 显示推荐结果
    if not recommended_heroes_df.empty:
        st.write('推荐英雄及其分路：')
        for index, row in recommended_heroes_df.iterrows():
            hero_name = row['hero_name']
            hero_role = row['role']
            st.write(f"英雄：{hero_name}, 分路：{hero_role}")
            image_extensions = ['.jpg', '.png']  # 定义可能的图片后缀
            # 这里需要确保图片路径正确，并且文件存在
            image_path_found = False  # 用于标记是否找到了图片路径
            for ext in image_extensions:
                image_path = rf'C:\Users\田六六\Desktop\王者荣耀英雄.zip\{hero_name}{ext}'
                if os.path.exists(image_path):  # 检查文件是否存在
                    st.image(image_path, use_column_width=True)
                    image_path_found = True
                    break  # 找到图片后退出循环
            if not image_path_found:  # 如果所有尝试的后缀都没有找到图片
                st.write(f"无法找到{hero_name}的图片。")       
        user_score = st.number_input('给推荐的英雄评分 (1-10):', min_value=1, max_value=10, value=5)
        submit_button = st.button('提交评分')
        if submit_button:
            st.write('评分已提交！')
    else:
        st.write('没有找到符合您偏好的英雄。')

if __name__ == '__main__':
    main()