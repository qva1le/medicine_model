import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# 1. Загрузка данных
df = pd.read_csv(r'C:\Users\Лёва\Desktop\Disease_symptom_and_patient_profile_dataset.csv')

# Выведем информацию о типах данных
print("Типы данных в датасете:")
print(df.dtypes)
print("\nКатегориальные столбцы:")
print(df.select_dtypes(include=['object']).columns.tolist())

# 2. Преобразуем все категориальные столбцы в числовые с помощью one-hot encoding
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
if 'Outcome Variable' in categorical_columns:
    categorical_columns.remove('Outcome Variable')

df_encoded = pd.get_dummies(df, columns=categorical_columns)

# 3. Преобразуем целевую переменную в числовую
label_encoder = LabelEncoder()
df_encoded['Outcome Variable'] = label_encoder.fit_transform(df_encoded['Outcome Variable'])

# 4. Разделение на признаки (X) и целевую переменную (y)
X = df_encoded.drop('Outcome Variable', axis=1)
y = df_encoded['Outcome Variable']

print("\nТипы данных после преобразования:")
print(X.dtypes)

# 5. Разделение на тренировочные и тестовые выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Создание и обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Прогнозирование и оценка модели
y_pred = model.predict(X_test)

# 8. Расчет и вывод различных метрик точности
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n===== ОЦЕНКА ТОЧНОСТИ МОДЕЛИ =====")
print(f'Accuracy (Точность): {accuracy:.4f}')
print(f'Precision (Точность положительного прогноза): {precision:.4f}')
print(f'Recall (Полнота): {recall:.4f}')
print(f'F1-score (F-мера): {f1:.4f}')

# 9. Вывод матрицы ошибок
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nМатрица ошибок:")
print(conf_matrix)

# 10. Вывод подробного отчета о классификации
print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 11. Кросс-валидация для более надежной оценки
cv_scores = cross_val_score(model, X, y, cv=5)
print("\nРезультаты кросс-валидации (5-fold):")
print(f"Средняя точность: {cv_scores.mean():.4f}")
print(f"Стандартное отклонение: {cv_scores.std():.4f}")
print(f"Все оценки: {cv_scores}")

# 12. Функция для ввода данных нового пациента
def get_patient_data():
    new_patient = pd.DataFrame(0, index=[0], columns=X.columns)

    # Ввод числовых данных
    new_patient['Age'] = int(input("Введите возраст пациента: "))

    # Ввод категориальных данных
    fever = input("Есть ли у пациента лихорадка? (да/нет): ").lower()
    new_patient['Fever_Yes'] = 1 if fever == 'да' else 0
    new_patient['Fever_No'] = 0 if fever == 'да' else 1

    cough = input("Есть ли у пациента кашель? (да/нет): ").lower()
    new_patient['Cough_Yes'] = 1 if cough == 'да' else 0
    new_patient['Cough_No'] = 0 if cough == 'да' else 1

    fatigue = input("Чувствует ли пациент усталость? (да/нет): ").lower()
    new_patient['Fatigue_Yes'] = 1 if fatigue == 'да' else 0
    new_patient['Fatigue_No'] = 0 if fatigue == 'да' else 1

    breathing = input("Есть ли у пациента затрудненное дыхание? (да/нет): ").lower()
    new_patient['Difficulty Breathing_Yes'] = 1 if breathing == 'да' else 0
    new_patient['Difficulty Breathing_No'] = 0 if breathing == 'да' else 1

    gender = input("Пол пациента (мужской/женский): ").lower()
    new_patient['Gender_Male'] = 1 if gender == 'мужской' else 0
    new_patient['Gender_Female'] = 1 if gender == 'женский' else 0

    bp = input("Кровяное давление пациента (высокое/низкое/нормальное): ").lower()
    new_patient['Blood Pressure_High'] = 1 if bp == 'высокое' else 0
    new_patient['Blood Pressure_Low'] = 1 if bp == 'низкое' else 0
    new_patient['Blood Pressure_Normal'] = 1 if bp == 'нормальное' else 0

    cholesterol = input("Уровень холестерина пациента (высокий/нормальный): ").lower()
    new_patient['Cholesterol Level_High'] = 1 if cholesterol == 'высокий' else 0
    new_patient['Cholesterol Level_Normal'] = 1 if cholesterol == 'нормальный' else 0

    return new_patient

# 13. Получаем данные пациента и делаем прогноз
print("\n===== ПРОГНОЗИРОВАНИЕ ДЛЯ НОВОГО ПАЦИЕНТА =====")
patient_data = get_patient_data()

# Прогноз для нового пациента
prediction = model.predict(patient_data)

# Проверяем, есть ли метод predict_proba
if hasattr(model, 'predict_proba'):
    prediction_proba = model.predict_proba(patient_data)

    # Вывод вероятностей для каждого класса
    print("\nВероятности для каждого класса:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"{class_name}: {prediction_proba[0][i]:.4f} ({prediction_proba[0][i]*100:.2f}%)")
    print(f'Прогноз для нового пациента: {label_encoder.inverse_transform(prediction)}')